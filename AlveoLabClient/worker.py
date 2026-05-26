"""Background-thread runner for algorithm calls.

Algorithms in `AlveoLab` are blocking and can take several seconds. Running
them on the GUI thread freezes the viewport, so `AlgorithmRunner` ferries the
call to a `QThread` and reports back via signals.

Usage from `MainWindow`:

    self._spawn_runner(
        algorithm,
        mesh,
        arch_type,
        on_done=self._on_segmentation_finished,
        on_fail=self._on_algorithm_failed,
    )

`_spawn_runner` keeps strong references to the worker + thread so they are not
garbage-collected mid-run; both are released via `deleteLater` once the run
ends.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal, Slot


class AlgorithmRunner(QObject):
    """Wrap a `(algorithm, *args, **kwargs)` call as a QObject worker."""

    started = Signal()
    finished = Signal(object)               # (result,)
    failed = Signal(object)                 # (exception,)

    def __init__(self, algorithm: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.algorithm = algorithm
        self.args = args
        self.kwargs = kwargs

    @Slot()
    def run(self) -> None:
        self.started.emit()
        try:
            result = self.algorithm.run(*self.args, **self.kwargs)
        except BaseException as exc:  # noqa: BLE001 - we forward to UI
            self.failed.emit(exc)
            return
        self.finished.emit(result)


def spawn_runner(
    owner: QObject,
    algorithm: Any,
    *args: Any,
    on_done: Callable[[Any], None],
    on_fail: Callable[[BaseException], None],
    on_started: Callable[[], None] | None = None,
    **kwargs: Any,
) -> tuple[QThread, AlgorithmRunner]:
    """Run `algorithm.run(*args, **kwargs)` on a fresh QThread.

    The thread and worker are parented to `owner` for the duration of the run
    and clean themselves up via `deleteLater` once `finished` or `failed`
    fires. Returns the thread and worker so the caller can keep additional
    strong references if it wants.
    """
    thread = QThread(owner)
    worker = AlgorithmRunner(algorithm, *args, **kwargs)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    if on_started is not None:
        worker.started.connect(on_started)

    def _done(result: Any) -> None:
        try:
            on_done(result)
        finally:
            thread.quit()

    def _fail(exc: BaseException) -> None:
        try:
            on_fail(exc)
        finally:
            thread.quit()

    worker.finished.connect(_done)
    worker.failed.connect(_fail)

    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker
