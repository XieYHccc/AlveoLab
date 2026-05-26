# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview
Alveolab aims to develop a clinical decision-support software system for cleft lip and palate treatment outcome assessment based on dental mesh data.

The core input of the system is a 3D dental mesh, which may be acquired from different scanners, clinical workflows, or coordinate systems. To ensure consistent downstream analysis, the system first performs dental model pose standardization, aligning meshes from different data sources into a unified reference pose.

On the standardized dental mesh, the system automatically performs tooth segmentation to identify and separate individual teeth. It then detects dental landmarks on the segmented teeth, locating clinically meaningful anatomical points required for treatment outcome assessment.

The segmented teeth and detected landmarks are passed as structured inputs to dedicated assessment algorithms, which automatically compute clinical scores or quantitative indicators for evaluating treatment outcomes in patients with cleft lip and palate.

The goal of the project is not to replace clinical judgment, but to provide an automated, objective, reproducible, and interpretable tool that supports clinicians in assessing treatment outcomes more efficiently and consistently.

## Tech Stack
The project is developed primarily in Python.

The core geometry processing and mesh manipulation are based on `trimesh`, which is used for loading, processing, analyzing, and transforming dental mesh data.

For visualization and the client-side user interface, the project uses `PyQt` together with `PyVista`. `PyQt` provides the desktop application framework and interactive UI components, while `PyVista` is used for rendering, inspecting, and interacting with 3D dental meshes, tooth segmentation results, dental landmarks, and assessment outputs.

The current core technology stack includes:

- **Programming language:** Python
- **Mesh processing:** `trimesh`
- **Desktop UI / client:** `PyQt`
- **3D visualization:** `PyVista`

## Environment & commands

A local virtualenv lives at `env/` (Windows layout — use `env/Scripts/python.exe`). All Python commands assume this interpreter unless you've activated it.

- Install for development: `env/Scripts/python.exe -m pip install -e .`
- Run all tests: `env/Scripts/python.exe -m pytest tests`
- Run a single test file: `env/Scripts/python.exe -m pytest tests/test_mhb_keypoints.py`
- Run a single test: `env/Scripts/python.exe -m pytest tests/test_mhb_keypoints.py::test_label_mapping_matches_expected_tooth_families`
- Run a module's manual `__main__` debug viewer (most files in `AlveoLab/` and `AlveoLab/mhb/` have one that loads a sample case and opens a PyVista window): `env/Scripts/python.exe -m AlveoLab.mhb.pipeline`

There is no configured linter, formatter, or CI in this repo. `README.md` is empty.

## Architecture

The code is organised to mirror the clinical pipeline described in the overview: **pose standardization → tooth segmentation → landmark detection → assessment**. Today only the first three stages have algorithmic code in this repo (assessment is consumed downstream); the modules below are arranged in that order.

### Foundations — depended on by everything else

- `AlveoLab.mesh.Mesh` (`mesh.py`): thin wrapper around `trimesh.Trimesh` that proxies attribute access (`__getattr__`) and **caches expensive geometry** via `utils.LazyAttribute` — face neighbours, edge / mean / gaussian / minimum / normal-variation curvatures. Construct with `Mesh.from_file(path)` or `Mesh.from_vertices_faces(...)`. Most downstream modules accept either a raw `trimesh.Trimesh` or a `Mesh`; prefer `Mesh` whenever curvature is needed, because results are memoised between recognizers.
- `AlveoLab.math` (`math/geometry.py`, `least_square_quadratic.py`, `oriented_bounding_box.py`, `kdtree.py`): pure-numpy geometry primitives (vector normalisation, projections, weighted quadratic fits, OBB, KD-trees). No mesh-specific assumptions; consumed by every higher layer.
- `AlveoLab.trimesh_utils`: low-level helpers built on top of `trimesh` (Laplacian / cotangent matrices, discrete curvature measures, face-face adjacency, local-maximum search). This is the seam where the `trimesh` dependency is concentrated — algorithm code calls through these helpers rather than touching `trimesh` directly.
- `AlveoLab.utils`: cross-cutting infrastructure — `LazyAttribute` (caching descriptor used by `Mesh` and elsewhere), `get_logger`, `load_labels`, `infer_arch_type`, plus small functional helpers (`mask_or`, `cached`, …). No domain logic; safe to import from anywhere.
- `AlveoLab.peak.Peak`: vendored from the ALR project (GPL-3.0). Do not relicense; keep the file header.

### Stage 1 — pose standardization (`AlveoLab/orienter/`)

Aligns a raw scan into the canonical dental frame (`right` / `forward` / `up`, plus `occlusal` derived from `arch_type ∈ {"U", "L"}`). All orienters subclass `BaseOrienter` (`_base_orienter.py`), which fixes the attribute contract: `center`, `right`, `forward`, `up`, `occlusal`, `axes`, `to_origin_transform_matrix`, plus `to_horizontal` / `from_horizontal` projection helpers.

| Class                              | Strategy                                                  | When to use                                                      |
|------------------------------------|-----------------------------------------------------------|-------------------------------------------------------------------|
| `PcaOrienter` (`_pca` + `pca_dental_orienter.py`) | PCA over face centres, axis-sign fix-ups from face normals / arch quadratic | Default for unlabeled meshes (neonatal pipeline)                  |
| `ObbOrienter` (`obb_dental_orienter.py`)         | Oriented bounding box from `trimesh.bounds.oriented_bounds` | When the cleft / gum dominates the mesh and PCA is noisy          |
| `ToothRegionPcaOrienter` (`mhb/orienter.py`)     | PCA on tooth-labelled vertices only, with full-mesh PCA fallback | Default for the MHB pipeline (per-vertex labels available)        |
| `ToothOrienter` (`tooth_orienter.py`)            | Per-tooth `{occlusal, distal, buccal}` from arch quadratic + OBB | Inside the older segmentation pipeline, per-tooth processing      |

Every later stage takes orientation as an input — never re-derives it — so consistency comes from passing the same orienter / `GlobalFrame` through the pipeline.

### Stage 2 — tooth segmentation (`AlveoLab/segmentation/`)

Multiple strategies coexist; they all produce per-vertex / per-face label arrays via `segmentation.label_arrays.build_face_and_vertex_label_arrays`:

- `curvature_based_seg.CurvatureBasedSeg` — peak-based grouping of overlapping high-curvature areas. Pulls `Peak`, `Tooth`, `OverlappingAreaGroup`, `Grouping`, `Quadratic3D`, and an orienter together; this is the core of the older neonatal pipeline.
- `harmonic_based_seg.HarmonicBasedSeg` — solves a cotangent-Laplacian harmonic field over the gum (`trimesh_utils.get_cotangent_weights_laplacian_matrix`, `scipy.sparse.linalg`), then extracts isoline loops per tooth via `isoline_voting` and crops with `cutting.find_optimal_gingiva_plane_trimesh_mean_paperlike`.
- `isoline_voting.py` — marching-triangles isoline extraction + loop candidate scoring, used by the harmonic strategy.
- `extract_tooth_from_boundary.py` — given a candidate boundary loop, grow / clean a single-tooth submesh.
- `tooth.Tooth`, `overlapping_area_group.OverlappingAreaGroup` — domain objects representing one tooth and the connected curvature regions that compose it.

`evaluate_segmentation.py` (top-level script) takes ground-truth + predicted label JSONs and reports per-tooth metrics; it’s the standard entry point for comparing segmentation backends.

### Stage 3 — landmark / keypoint detection

Two pipelines exist for historical reasons. Both consume `(mesh, arch_type[, vertex_labels])` and produce keypoints in world coordinates, but they target different patient ages and use different abstractions.

**3a. Neonatal pipeline — `AlveoLab.landmark_recognizer.LandmarkRecognizer`**
End-to-end on an unlabelled arch: pick an orienter (PCA or OBB) → run `HarmonicBasedSeg` / `CurvatureBasedSeg` to find tooth peaks → classify peaks as L/R landmarks against the arch quadratic. `cleft_classifier.CleftClassifier` is a sibling consumer that takes the same OBB-aligned mesh and labels the cleft as **unilateral (L/R) vs bilateral** and **complete vs incomplete**.

**3b. MHB keypoint pipeline — `AlveoLab/mhb/`**
The newer, more modular pipeline for 5-year-old dentitions where per-vertex tooth labels are already available. Read `mhb/__init__.py` first for the public surface. Flow inside `MhbKeypointPipeline` (`mhb/pipeline.py`):

1. **`GlobalFrame.from_mesh`** (`mhb/frame.py`) — wraps `ToothRegionPcaOrienter` (or a caller-supplied orienter) into an immutable `{right, forward, occlusal, center}` dataclass that every later step reads from.
2. **`build_arch_medial_curve`** (`mhb/medial_curve.py`) — fits an `ArchMedialCurve` across labelled teeth in the occlusal plane; later used to derive each tooth’s mesiodistal/buccolingual axes.
3. **`build_tooth_context`** (`mhb/context.py`) — for each label in `TOOTH_LABEL_DEFINITIONS` (12 teeth, ids 1–12, see `mhb/labels.py`), extracts the per-tooth submesh + local axes into a `ToothContext` (mesh, vertex indices, mesiodistal/buccolingual/occlusal axes and projected values, boundary mask, horizontal scale, axis source). Optionally keeps only the largest connected component.
4. **`ToothRecognizerRegistry.resolve(context.definition)`** (`mhb/registry.py`) — dispatches by `definition.family` (`INCISOR`, `CANINE`, `PRIMARY_MOLAR`, `MOLAR`) to a `BaseToothKeypointRecognizer` subclass under `mhb/recognizers/`. Per-label overrides are also supported (`register_label`). Each recognizer returns a `ToothKeypointResult` with typed `MhbKeypoint`s (kind, 3D point, source vertex index, score, metadata).

Shared utilities used by recognizers: `mhb/cusp_detection.py` (`detect_cusps_local_extrema`, `detect_cusps_watershed`, `watershed_basins`, `merge_spurious_basins`, `compute_vertex_curvature`) and `mhb/cross_section.py` (`build_tooth_cross_sections`).

**When adding a new tooth family or changing keypoint logic**: add/edit a recognizer under `mhb/recognizers/` (subclass `BaseToothKeypointRecognizer`, use `self._make_keypoint` / `self._result`), register it in `mhb/registry.py`, and extend `TOOTH_LABEL_DEFINITIONS` only if the family/label mapping itself changes.

### Visualization & UI seam (`pyvista_utils.py`, `visualization.py`, the `__main__` blocks)

These are **presentation only** — they take a `Mesh` / orienter / `MhbKeypointPipeline` and build a `pyvista.Plotter`. Keep all algorithmic APIs returning numpy arrays / dataclasses so the planned PyQt client can embed a `pyvistaqt.QtInteractor` and call the same builders without invoking `plotter.show()`. Do **not** import PyQt or PyVista from algorithm modules; the dependency arrow always points *toward* visualization, never away from it.

### Convention: `arch_type`
Throughout the codebase `arch_type` is the single-letter string `"U"` (upper / maxillary) or `"L"` (lower / mandibular). `utils.infer_arch_type` parses it from filenames containing `upper` / `maxillary` or `lower` / `mandibular`.

### Lazy curvature pattern
Tests often inject precomputed curvature by setting the private attribute directly on a `Mesh` (e.g. `mesh._vertex_mean_curvature = curvature`) — this works because `LazyAttribute` stores under the same underscore name. Preserve that pattern when adding new lazy geometry attributes.
