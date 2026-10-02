# Join the 50 partial grid-search CSVs into one table.
import sys
from pathlib import Path

import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from runner_file.grid_search_runner import N_JOBS, N_TOTAL, OUTPUT_DIR, PARTS_DIR

COMBINED_PATH = OUTPUT_DIR / "grid_search.csv"


def combine() -> pd.DataFrame:
    frames = []
    missing = []
    for job_id in range(N_JOBS):
        path = PARTS_DIR / f"job_{job_id:02d}.csv"
        if not path.exists():
            missing.append(path.name)
            continue
        frames.append(pd.read_csv(path))
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} partial files in {PARTS_DIR}: {', '.join(missing)}"
        )

    combined = pd.concat(frames, ignore_index=True).sort_values("point_index")
    if combined["point_index"].nunique() != N_TOTAL or len(combined) != N_TOTAL:
        raise ValueError(
            f"Expected {N_TOTAL} unique points, found {len(combined)} rows "
            f"and {combined['point_index'].nunique()} unique point_index values."
        )
    if not bool(combined.loc[combined["point_index"] == 0, "is_ground_truth"].iloc[0]):
        raise ValueError("Point 0 is not marked as the ground truth.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_PATH, index=False)
    print(f"Wrote {len(combined)} rows to {COMBINED_PATH}")
    return combined


if __name__ == "__main__":
    combine()
