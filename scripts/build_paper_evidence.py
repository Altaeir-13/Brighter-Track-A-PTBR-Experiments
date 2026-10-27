from pathlib import Path

from brighter.paper_evidence import generate_paper_package


if __name__ == "__main__":
    result = generate_paper_package(Path(__file__).resolve().parents[1])
    print(f"paper evidence frozen: {result['master_rows']} systems, {result['claims']} claims")
