from pathlib import Path

from brighter.threshold_reporting import generate_threshold_reports

if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]
    result=generate_threshold_reports(root)
    for row in result["main_results"]:
        print(f"{row['system']}: {row['fixed_macro_f1']:.4f} -> {row['thresholded_macro_f1']:.4f} ({row['delta_macro_f1']:+.4f})")
    print("Test set: SEALED - NOT EVALUATED")
