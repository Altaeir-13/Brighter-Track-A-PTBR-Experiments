from pathlib import Path

from brighter.encoder_reporting import generate_encoder_reports


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    summary = generate_encoder_reports(project_root)
    for row in summary["main_results"]:
        if row["model"].startswith("E"):
            print(f"{row['model']}: Macro-F1={row['macro_f1']:.4f}")
    print("Test set: SEALED - NOT EVALUATED")
