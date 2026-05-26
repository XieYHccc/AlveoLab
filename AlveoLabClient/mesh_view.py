"""3D viewport widget that subscribes to `SceneModel` and renders three layers.

Layer 1: the base mesh (white).
Layer 2: per-tooth coloring overlaid on the base mesh via cell scalars.
Layer 3: red spheres for landmark peaks.

Each layer is owned by its own actor (or actor list) so an update to one does
not invalidate the others. This is what lets segmentation + landmarks be shown
at the same time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv
from pyvista import plotting
from pyvistaqt import QtInteractor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from AlveoLab.pyvista_utils import mesh_to_polydata

from .algorithms import LandmarkResult, SegmentationResult
from .palette import PALETTE, palette_for_labels
from .scene import SceneModel


class MeshView(QWidget):
    """A QWidget embedding `pyvistaqt.QtInteractor`.

    Connect to a `SceneModel` after construction:

        view = MeshView(parent)
        view.bind(scene)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Match the look of `AlveoLab.pyvista_utils.get_dental_plotter`:
        # DocumentTheme, default lights disabled, a single camera light. Without
        # this the mesh shows up as a flat white silhouette with no 3D cue.
        theme = plotting.themes.DocumentTheme()
        self.plotter = QtInteractor(self, theme=theme, lighting="none")
        self.plotter.enable_anti_aliasing()
        self._install_lighting()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter.interactor)

        # Layer state ------------------------------------------------------
        self._mesh: Any = None
        self._polydata: pv.PolyData | None = None
        self._mesh_actor: Any = None
        self._landmark_actors: list[Any] = []
        self._mesh_diag: float = 0.0

    # ---- scene binding ----------------------------------------------------

    def bind(self, scene: SceneModel) -> None:
        scene.mesh_loaded.connect(self._on_mesh_loaded)
        scene.segmentation_updated.connect(self._on_segmentation_updated)
        scene.landmarks_updated.connect(self._on_landmarks_updated)
        scene.cleared.connect(self._on_cleared)

    # ---- Layer 1: base mesh ----------------------------------------------

    def _on_mesh_loaded(self, mesh: Any) -> None:
        self._clear_all_actors()
        self._install_mesh_actor(mesh, reset_camera=True)

    def _install_mesh_actor(self, mesh: Any, *, reset_camera: bool) -> None:
        """Swap the base mesh actor to render `mesh`.

        Re-used by initial load and by segmentation when the algorithm
        returned its own (typically gum-cropped) mesh. Drops any previous
        mesh actor first; landmark actors are left alone.
        """
        if self._mesh_actor is not None:
            try:
                self.plotter.remove_actor(self._mesh_actor)
            except Exception:  # noqa: BLE001
                pass
            self._mesh_actor = None

        self._mesh = mesh
        self._polydata = mesh_to_polydata(mesh)

        vertices = np.asarray(mesh.vertices, dtype=float)
        if vertices.size:
            self._mesh_diag = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        else:
            self._mesh_diag = 0.0

        # Initialize per-face RGB to gum so segmentation can recolor in place
        # via cell_data updates.
        n_faces = int(np.asarray(mesh.faces).shape[0])
        gum_rgb = np.tile(PALETTE[0], (n_faces, 1))
        self._polydata.cell_data["tooth_rgb"] = gum_rgb

        # Material values match `AlveoLab.mhb.pipeline` __main__ and
        # `LandmarkRecognizerVisualization.add_mesh_with_labels`.
        self._mesh_actor = self.plotter.add_mesh(
            self._polydata,
            scalars="tooth_rgb",
            rgb=True,
            opacity=1.0,
            specular=0.0,
            specular_power=5,
            ambient=0.2,
            show_edges=False,
            name="base_mesh",
        )

        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()

    # ---- Layer 2: segmentation coloring ----------------------------------

    def _on_segmentation_updated(self, result: SegmentationResult) -> None:
        # If the algorithm produced its own (typically gum-cropped) mesh, swap
        # the viewport to render it. Labels refer to that mesh, not the input.
        if result.mesh is not None and result.mesh is not self._mesh:
            self._install_mesh_actor(result.mesh, reset_camera=False)

        if self._polydata is None or self._mesh is None:
            return

        # Prefer face labels (cleaner cell coloring). Fall back to vertex
        # labels by per-face vertex vote if face labels are missing.
        if result.face_labels is not None:
            face_labels = np.asarray(result.face_labels, dtype=np.int64)
        else:
            faces = np.asarray(self._mesh.faces, dtype=np.int64)
            vlabels = np.asarray(result.vertex_labels, dtype=np.int64)
            face_vlabels = vlabels[faces]
            face_labels = np.where(
                face_vlabels.max(axis=1) > 0,
                face_vlabels.max(axis=1),
                0,
            )

        n_faces = int(np.asarray(self._mesh.faces).shape[0])
        if face_labels.shape[0] != n_faces:
            # Shape mismatch should not happen once `result.mesh` is honored —
            # but if some future algorithm forgets to set it, keep showing the
            # base mesh in gum colour rather than rendering bogus colors.
            face_labels = np.zeros(n_faces, dtype=np.int64)

        rgb = palette_for_labels(face_labels)
        self._polydata.cell_data["tooth_rgb"] = rgb
        # cell_data assignment replaces the underlying array; tell VTK the
        # polydata has changed so the mapper rebuilds the cell colours.
        self._polydata.Modified()
        self.plotter.render()

    # ---- Layer 3: landmark spheres ---------------------------------------

    def _on_landmarks_updated(self, result: LandmarkResult) -> None:
        # Drop previous landmark actors.
        for actor in self._landmark_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:  # noqa: BLE001
                pass
        self._landmark_actors.clear()

        if not result.landmarks:
            self.plotter.render()
            return

        radius = max(0.4, 0.012 * self._mesh_diag) if self._mesh_diag else 0.6
        color = (0.90, 0.20, 0.20)

        for i, lm in enumerate(result.landmarks):
            point = np.asarray(lm.point, dtype=float)
            sphere = pv.Sphere(radius=radius, center=point)
            actor = self.plotter.add_mesh(
                sphere,
                color=color,
                opacity=0.92,
                specular=0.4,
                specular_power=20,
                ambient=0.3,
                name=f"landmark_{i}",
            )
            self._landmark_actors.append(actor)

        self.plotter.render()

    # ---- clear -----------------------------------------------------------

    def _on_cleared(self) -> None:
        self._clear_all_actors()
        self.plotter.render()

    def _clear_all_actors(self) -> None:
        for actor in self._landmark_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:  # noqa: BLE001
                pass
        self._landmark_actors.clear()

        if self._mesh_actor is not None:
            try:
                self.plotter.remove_actor(self._mesh_actor)
            except Exception:  # noqa: BLE001
                pass
            self._mesh_actor = None

        # Defensive: drop anything else still in the renderer. Use
        # `clear_actors` (not `plotter.clear`!) — the latter calls
        # `remove_all_lights`, which would wipe our camera light and leave the
        # next mesh unshaded.
        try:
            self.plotter.renderer.clear_actors()
        except Exception:  # noqa: BLE001
            pass

        # Re-add the camera light if something upstream removed it.
        self._install_lighting()

        self._mesh = None
        self._polydata = None
        self._mesh_diag = 0.0

    def _install_lighting(self) -> None:
        """Install a single camera-following white light if none exists.

        Matches the lighting installed by
        `AlveoLab.pyvista_utils.get_dental_plotter`. Safe to call multiple
        times; only installs the light when the renderer has no lights left.
        """
        try:
            n_lights = self.plotter.renderer.GetLights().GetNumberOfItems()
        except Exception:  # noqa: BLE001
            n_lights = 0
        if n_lights >= 1:
            return
        self._camera_light = pv.Light(
            color="white", light_type="camera light", intensity=0.8
        )
        self.plotter.add_light(self._camera_light)

    # ---- cleanup ---------------------------------------------------------

    def closeEvent(self, event):  # type: ignore[override]
        try:
            self.plotter.close()
        finally:
            super().closeEvent(event)
