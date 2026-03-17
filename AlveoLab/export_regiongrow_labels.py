import json
from pathlib import Path

from AlveoLab.landmark_recognizer import LandmarkRecognizer
from AlveoLab.mesh import Mesh

def infer_arch_type(obj_path: Path) -> str:
    name = obj_path.stem.lower()
    if "upper" in name or "maxillary" in name:
        return "U"
    if "lower" in name or "mandibular" in name:
        return "L"
    raise ValueError(f"Cannot infer arch type from file name: {obj_path.name}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    input_dir = repo_root / "data" / "labeld_5year_betterv_objs"
    output_dir = repo_root / "saved" / "hf"
    output_dir.mkdir(parents=True, exist_ok=True)

    failed_objs = []

    for obj_path in sorted(input_dir.glob("*.obj")):
        try:
            arch_type = infer_arch_type(obj_path)
            mesh = Mesh.from_file(obj_path)
            lr = LandmarkRecognizer(mesh, arch_type)
            hf = lr.harmonic_field
            output_path = output_dir / f"{obj_path.stem}.json"
            payload = {
                "patient_id": obj_path.stem,
                "labels": lr.harmonic_seg.harmonic_vertex_labels.tolist(),
                # "labels": lr.teeth_vertex_labels.tolist(),
            }
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            failed_objs.append(obj_path.name)

    if failed_objs:
        print("Failed OBJ files:")
        for name in failed_objs:
            print(name)
    else:
        print("All OBJ files processed successfully.")


if __name__ == "__main__":
    main()
