"""Application shell: menus, central `MeshView`, status bar, and action wiring.

Menus are built from an :class:`AlgorithmRegistry` so adding new methods does
not require touching this file:

* Categories with a single registered algorithm render as a single menu item.
* Categories with multiple algorithms render as a submenu listing each name.
* Empty categories are skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStatusBar,
)

from AlveoLab.mesh import Mesh
from AlveoLab.utils import get_logger, infer_arch_type

from .algorithms import (
    AlgorithmRegistry,
    LandmarkAlgorithm,
    LandmarkResult,
    SegmentationAlgorithm,
    SegmentationResult,
    default_registry,
)
from .dialogs import ArchTypeDialog, show_error
from .mesh_view import MeshView
from .scene import SceneModel
from .worker import spawn_runner


logger = get_logger("AlveoLabClient")

MESH_FILE_FILTER = "Dental meshes (*.obj *.stl *.ply);;All files (*)"


class MainWindow(QMainWindow):
    def __init__(self, registry: AlgorithmRegistry | None = None) -> None:
        super().__init__()
        self.setWindowTitle("AlveoLab Client")
        self.resize(1280, 860)

        self.scene = SceneModel(self)
        self.registry = registry or default_registry()

        self.view = MeshView(self)
        self.view.bind(self.scene)
        self.setCentralWidget(self.view)

        self.setStatusBar(QStatusBar(self))
        self._set_status("Ready")

        self._tool_actions: list[QAction] = []
        self._build_menus()

        # Keep strong refs to active worker threads/runners so they aren't GC'd.
        self._workers: list[tuple[Any, Any]] = []

    # ---- menu construction -----------------------------------------------

    def _build_menus(self) -> None:
        menubar = self.menuBar()

        # File menu --------------------------------------------------------
        file_menu = menubar.addMenu("&File")

        open_action = QAction("Open mesh…", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open_mesh)
        file_menu.addAction(open_action)

        close_action = QAction("Close mesh", self)
        close_action.triggered.connect(self._on_close_mesh)
        file_menu.addAction(close_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Tools menu -------------------------------------------------------
        tools_menu = menubar.addMenu("&Tools")
        self._add_category(
            tools_menu,
            label="Segment teeth",
            entries=self.registry.segmentation,
            handler=self._run_segmentation,
        )
        self._add_category(
            tools_menu,
            label="Detect landmarks",
            entries=self.registry.landmark,
            handler=self._run_landmark,
        )
        if self.registry.assessment:
            self._add_category(
                tools_menu,
                label="Assessment",
                entries=self.registry.assessment,
                handler=self._run_assessment,
            )

    def _add_category(
        self,
        parent_menu: QMenu,
        label: str,
        entries: dict,
        handler,
    ) -> None:
        if not entries:
            return

        if len(entries) == 1:
            name, algorithm = next(iter(entries.items()))
            action = QAction(f"{label} ({name})", self)
            action.triggered.connect(lambda checked=False, a=algorithm: handler(a))
            parent_menu.addAction(action)
            self._tool_actions.append(action)
            return

        submenu = parent_menu.addMenu(label)
        for name, algorithm in entries.items():
            action = QAction(name, self)
            action.triggered.connect(lambda checked=False, a=algorithm: handler(a))
            submenu.addAction(action)
            self._tool_actions.append(action)

    # ---- File actions ----------------------------------------------------

    def _on_open_mesh(self) -> None:
        start_dir = str(Path.cwd())
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open dental mesh", start_dir, MESH_FILE_FILTER
        )
        if not path_str:
            return
        path = Path(path_str)

        try:
            mesh = Mesh.from_file(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load mesh: %s", path)
            self._set_status("Load failed")
            show_error(self, "Failed to load mesh", exc)
            return

        try:
            arch_type = infer_arch_type(path)
        except ValueError:
            arch_type = self._prompt_arch_type(path.name)
            if arch_type is None:
                self._set_status("Load cancelled")
                return

        self.scene.set_mesh(mesh, arch_type, source_path=path)
        self._set_status(f"Loaded {path.name} — arch {arch_type}")

    def _on_close_mesh(self) -> None:
        if not self.scene.has_mesh:
            return
        self.scene.clear()
        self._set_status("Closed mesh")

    def _prompt_arch_type(self, filename: str) -> str | None:
        dlg = ArchTypeDialog(filename, parent=self)
        if dlg.exec() == ArchTypeDialog.Accepted:
            return dlg.selected_arch_type()
        return None

    # ---- Tools actions ---------------------------------------------------

    def _run_segmentation(self, algorithm: SegmentationAlgorithm) -> None:
        if not self._require_mesh("segment teeth"):
            return
        self._set_tools_enabled(False)
        self._set_status(f"Running segmentation ({algorithm.name})…")
        self._spawn(
            algorithm,
            self.scene.mesh,
            self.scene.arch_type,
            on_done=self._on_segmentation_done,
            on_fail=lambda exc: self._on_algorithm_failed(exc, kind="Segmentation"),
        )

    def _run_landmark(self, algorithm: LandmarkAlgorithm) -> None:
        if not self._require_mesh("detect landmarks"):
            return
        self._set_tools_enabled(False)
        self._set_status(f"Detecting landmarks ({algorithm.name})…")
        self._spawn(
            algorithm,
            self.scene.mesh,
            self.scene.arch_type,
            on_done=self._on_landmark_done,
            on_fail=lambda exc: self._on_algorithm_failed(exc, kind="Landmark detection"),
        )

    def _run_assessment(self, algorithm: Any) -> None:
        # Placeholder hook for v2. Kept here so the menu wiring is uniform.
        QMessageBox.information(
            self,
            "Assessment",
            f"Assessment algorithm '{algorithm.name}' is not wired up yet.",
        )

    def _on_segmentation_done(self, result: SegmentationResult) -> None:
        self.scene.set_segmentation(result)
        n = result.num_teeth
        self._set_status(f"Segmented {n} teeth")
        self._set_tools_enabled(True)

    def _on_landmark_done(self, result: LandmarkResult) -> None:
        self.scene.set_landmarks(result)
        n = len(result.landmarks)
        self._set_status(f"Detected {n} landmarks")
        self._set_tools_enabled(True)

    def _on_algorithm_failed(self, exc: BaseException, kind: str) -> None:
        logger.exception("%s failed", kind, exc_info=exc)
        self._set_status(f"{kind} failed")
        show_error(self, f"{kind} failed", exc)
        self._set_tools_enabled(True)

    # ---- helpers ---------------------------------------------------------

    def _require_mesh(self, action_label: str) -> bool:
        if self.scene.has_mesh:
            return True
        QMessageBox.information(
            self,
            "No mesh loaded",
            f"Load a mesh from File → Open before trying to {action_label}.",
        )
        return False

    def _set_tools_enabled(self, enabled: bool) -> None:
        for action in self._tool_actions:
            action.setEnabled(enabled)

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def _spawn(self, algorithm, *args, on_done, on_fail, **kwargs) -> None:
        pair = spawn_runner(
            self,
            algorithm,
            *args,
            on_done=on_done,
            on_fail=on_fail,
            **kwargs,
        )
        self._workers.append(pair)
        # Prune finished workers — Qt has already torn them down at this point,
        # we just drop our strong references so the list does not grow forever.
        self._workers = [
            (t, w) for (t, w) in self._workers if not _is_thread_finished(t)
        ]


def _is_thread_finished(thread) -> bool:
    try:
        return thread.isFinished()
    except RuntimeError:
        # Underlying QThread already deleted by Qt.
        return True
