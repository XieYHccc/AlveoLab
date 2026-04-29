import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from AlveoLab.mesh import Mesh
from AlveoLab.trimesh_utils import get_oriented_bounding_box
from AlveoLab.utils import load_labels
EPS = 1e-8

def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    default_gt_dir = repo_root / "data" / "labeld_5year_betterv_objs"
    default_pred_dir = repo_root / "saved" / "pred_labels_tg_noalign"
    default_saved_dir = repo_root / "saved"/ "result_tg_noalign.txt"
    test_list = repo_root / "saved" / "val_list_new.txt"

    parser = argparse.ArgumentParser(
        description="Evaluate tooth segmentation quality from OBJ meshes and vertex labels."
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=default_gt_dir,
        help="Directory containing OBJ meshes and ground-truth JSON files.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=default_pred_dir,
        help="Directory containing predicted JSON files.",
    )
    parser.add_argument(
        "--list-file",
        type=Path,
        default=test_list,
        help="Optional TXT file. Each line is an OBJ file name to evaluate.",
    )
    parser.add_argument(
        "--pred-suffix",
        type=str,
        default="",
        help="Prediction JSON name suffix before .json, e.g. foo_regiongrow.json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_saved_dir,
        help="Optional path to save the evaluation summary and per-case results.",
    )
    return parser.parse_args()


def read_case_names(gt_dir: Path, list_file: Path | None) -> list[str]:
    if list_file is None:
        return sorted(path.name for path in gt_dir.glob("*.obj"))

    case_names = []
    with list_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if not line.lower().endswith(".obj"):
                line = f"{line}.obj"
            case_names.append(line)
    return case_names

def validate_label_length(case_name: str, vertex_count: int, labels: np.ndarray, label_type: str):
    if labels.ndim != 1:
        raise ValueError(f"{label_type} labels of {case_name} must be a 1D array.")
    if labels.shape[0] != vertex_count:
        raise ValueError(
            f"{label_type} labels of {case_name} have length {labels.shape[0]}, "
            f"but mesh has {vertex_count} vertices."
        )


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a, dtype=bool).reshape(-1)
    mask_b = np.asarray(mask_b, dtype=bool).reshape(-1)
    if mask_a.shape != mask_b.shape:
        raise ValueError("Dice masks must have the same shape.")

    intersection = np.count_nonzero(mask_a & mask_b)
    size_sum = np.count_nonzero(mask_a) + np.count_nonzero(mask_b)
    if size_sum == 0:
        return 1.0
    return 2.0 * intersection / size_sum


def compute_binary_tooth_gingiva_dsc(gt_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    return dice_coefficient(gt_labels > 0, pred_labels > 0)


def compute_oriented_box_center_and_diagonal(points: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3).")
    if points.shape[0] == 0:
        raise ValueError("Cannot compute tooth center from empty points.")

    if points.shape[0] == 1:
        return points[0], EPS

    obb = get_oriented_bounding_box(points)
    center = np.asarray(obb.center, dtype=np.float64)
    diagonal = max(float(np.linalg.norm(obb.extents)), EPS)
    return center, diagonal


def collect_tooth_geometry(vertices: np.ndarray, labels: np.ndarray) -> dict[int, dict[str, np.ndarray | float]]:
    tooth_info = {}
    for label in np.unique(labels):
        if label <= 0:
            continue
        tooth_points = vertices[labels == label]
        if tooth_points.shape[0] == 0:
            continue
        center, scale = compute_oriented_box_center_and_diagonal(tooth_points)
        tooth_info[int(label)] = {
            "center": center,
            "scale": scale,
        }
    return tooth_info


def compute_tooth_localization_scores(
    vertices: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> list[float]:
    gt_teeth = collect_tooth_geometry(vertices, gt_labels)
    pred_teeth = collect_tooth_geometry(vertices, pred_labels)

    if not gt_teeth:
        return []
    if not pred_teeth:
        return [0.0 for _ in gt_teeth]

    pred_centers = np.asarray([info["center"] for info in pred_teeth.values()], dtype=np.float64)
    scores = []

    for gt_info in gt_teeth.values():
        gt_center = np.asarray(gt_info["center"], dtype=np.float64)
        gt_scale = max(float(gt_info["scale"]), EPS)
        distances = np.linalg.norm(pred_centers - gt_center, axis=1)
        nearest_distance = float(np.min(distances))
        normalized_distance = nearest_distance / gt_scale
        scores.append(math.exp(-normalized_distance))

    return scores


def compute_teeth_labeling_scores(gt_labels: np.ndarray, pred_labels: np.ndarray) -> list[float]:
    gt_tooth_labels = [int(label) for label in np.unique(gt_labels) if label > 0]
    pred_tooth_labels = {int(label) for label in np.unique(pred_labels) if label > 0}

    scores = []
    for label in gt_tooth_labels:
        gt_mask = gt_labels == label
        if label not in pred_tooth_labels:
            scores.append(0.0)
            continue
        pred_mask = pred_labels == label
        scores.append(dice_coefficient(gt_mask, pred_mask))
    return scores


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(values))


def evaluate_case(obj_path: Path, gt_json_path: Path, pred_json_path: Path) -> dict:
    mesh = Mesh.from_file(obj_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    gt_labels = load_labels(gt_json_path, True)
    pred_labels = load_labels(pred_json_path, False)

    validate_label_length(obj_path.name, vertices.shape[0], gt_labels, "Ground-truth")
    validate_label_length(obj_path.name, vertices.shape[0], pred_labels, "Prediction")

    localization_scores = compute_tooth_localization_scores(vertices, gt_labels, pred_labels)
    labeling_scores = compute_teeth_labeling_scores(gt_labels, pred_labels)

    return {
        "patient_id": obj_path.stem,
        "obj_name": obj_path.name,
        "num_vertices": int(vertices.shape[0]),
        "num_gt_teeth": int(sum(label > 0 for label in np.unique(gt_labels))),
        "num_pred_teeth": int(sum(label > 0 for label in np.unique(pred_labels))),
        "tooth_gingiva_dsc": compute_binary_tooth_gingiva_dsc(gt_labels, pred_labels),
        "tooth_localization_accuracy": mean_or_zero(localization_scores),
        "teeth_labeling_accuracy": mean_or_zero(labeling_scores),
        "per_tooth_localization_scores": localization_scores,
        "per_tooth_labeling_scores": labeling_scores,
    }


def summarize_results(case_results: list[dict], failed_cases: list[str]) -> dict:
    pooled_localization_scores = []
    pooled_labeling_scores = []

    for result in case_results:
        pooled_localization_scores.extend(result["per_tooth_localization_scores"])
        pooled_labeling_scores.extend(result["per_tooth_labeling_scores"])

    return {
        "num_cases": len(case_results),
        "num_failed_cases": len(failed_cases),
        "failed_cases": failed_cases,
        "mean_case_tooth_gingiva_dsc": mean_or_zero(
            result["tooth_gingiva_dsc"] for result in case_results
        ),
        "mean_case_tooth_localization_accuracy": mean_or_zero(
            result["tooth_localization_accuracy"] for result in case_results
        ),
        "mean_case_teeth_labeling_accuracy": mean_or_zero(
            result["teeth_labeling_accuracy"] for result in case_results
        ),
        "mean_pooled_tooth_localization_accuracy": mean_or_zero(pooled_localization_scores),
        "mean_pooled_teeth_labeling_accuracy": mean_or_zero(pooled_labeling_scores),
    }


def print_summary(summary: dict, case_results: list[dict]):
    print("Evaluation Summary")
    print(f"Successful cases: {summary['num_cases']}")
    print(f"Failed cases: {summary['num_failed_cases']}")
    print(f"Mean case Tooth-gingiva DSC: {summary['mean_case_tooth_gingiva_dsc']:.6f}")
    print(
        "Mean case Tooth localization accuracy: "
        f"{summary['mean_case_tooth_localization_accuracy']:.6f}"
    )
    print(
        "Mean case Teeth labeling accuracy: "
        f"{summary['mean_case_teeth_labeling_accuracy']:.6f}"
    )
    print(
        "Mean pooled Tooth localization accuracy: "
        f"{summary['mean_pooled_tooth_localization_accuracy']:.6f}"
    )
    print(
        "Mean pooled Teeth labeling accuracy: "
        f"{summary['mean_pooled_teeth_labeling_accuracy']:.6f}"
    )

    if case_results:
        print("\nPer-case Results")
        for result in case_results:
            print(
                f"{result['obj_name']}: "
                f"DSC={result['tooth_gingiva_dsc']:.6f}, "
                f"Loc={result['tooth_localization_accuracy']:.6f}, "
                f"Label={result['teeth_labeling_accuracy']:.6f}"
            )

    if summary["failed_cases"]:
        print("\nFailed OBJ files:")
        for case_name in summary["failed_cases"]:
            print(case_name)


def main():
    args = parse_args()

    gt_dir = args.gt_dir.resolve()
    pred_dir = args.pred_dir.resolve()
    case_names = read_case_names(gt_dir, args.list_file)

    case_results = []
    failed_cases = []

    for case_name in case_names:
        obj_path = gt_dir / case_name
        gt_json_path = gt_dir / f"{obj_path.stem}.json"
        pred_json_path = pred_dir / f"{obj_path.stem}{args.pred_suffix}.json"

        try:
            if not obj_path.exists():
                raise FileNotFoundError(f"OBJ file not found: {obj_path}")
            if not gt_json_path.exists():
                raise FileNotFoundError(f"Ground-truth JSON not found: {gt_json_path}")
            if not pred_json_path.exists():
                raise FileNotFoundError(f"Prediction JSON not found: {pred_json_path}")

            case_results.append(evaluate_case(obj_path, gt_json_path, pred_json_path))
        except Exception as exc:
            failed_cases.append(case_name)
            print(f"Failed on {case_name}: {exc}")

    summary = summarize_results(case_results, failed_cases)
    print_summary(summary, case_results)

    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "cases": case_results,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
