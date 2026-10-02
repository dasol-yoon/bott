# Uniform sample of the simulated STO problem used by simulated_runner.py.
# Each Slurm job evaluates 10 of the 500 points and writes one partial CSV.
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from tifffile import imwrite

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from bott.loss import LossFunction
from bott.optimization import solve
from bott.physics_models import simulate_cbed
from bott.reduction import ReductionFunction
from bott.utils import print_system_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Match simulated_runner.py for GT (200, 3, -5), 3x3 square patches, and the
# epsilon interval [0.1, 10].
GROUND_TRUTH = (200.0, 3.0, -5.0)
BOUNDS = ((5.0, 500.0), (-10.0, 10.0), (-10.0, 10.0))
N_TOTAL = 500
N_PER_JOB = 10
N_JOBS = N_TOTAL // N_PER_JOB
SAMPLE_SEED = 42
OVERALL_SCALING_FACTOR = 1000
EPS_BASE = 10
EPS_C = 1
NUM_TILES = 3
# Same tile scale the optimizer applies: sqrt((150/3)^2) on every tile.
SCALE_FACTOR = (150 / NUM_TILES)
OUTPUT_DIR = _project_root / "results" / "grid_search" / "GT_200_3_-5"
PARTS_DIR = OUTPUT_DIR / "parts"
GROUND_TRUTH_PATH = OUTPUT_DIR / "ground_truth.tif"
POINTS_PATH = OUTPUT_DIR / "points.csv"

PARAMS_ABTEM = {
    "path_crystal": "/home/pb482/bott/data/SrTiO3.cif",
    "potential_extent_x": 62.6,
    "potential_extent_y": 62.6,
    "lateral_sampling": 0.2 * 2 / 3,
    "vertical_sampling": 2,
    "potential_parametrization": "lobato",
    "potential_projection": "finite",
    "random_seed": 42,
    "use_frozen_phonon": False,
    "num_phonon_configs": 5,
    "phonon_sigma": {"Sr": 0.088, "Ti": 0.0746, "O": 0.0963},
    "energy": 200e3,
    "convergence_angle": 19.1,
    "df": 0,
    "aberrations": {},
    "detector_angle": 31,
    "scan_step_size": 0.3,
    "return_pacbed": True,
}

PATCH_COLUMNS = [f"patch_{i}" for i in range(1, NUM_TILES ** 2 + 1)]
PATCH_TRUE_COLUMNS = [f"patch_true_{i}" for i in range(1, NUM_TILES ** 2 + 1)]


def sample_points() -> pd.DataFrame:
    """499 uniform points plus the ground truth, in a fixed order."""
    rng = np.random.default_rng(SAMPLE_SEED)
    samples = np.column_stack(
        [
            rng.uniform(BOUNDS[0][0], BOUNDS[0][1], N_TOTAL - 1),
            rng.uniform(BOUNDS[1][0], BOUNDS[1][1], N_TOTAL - 1),
            rng.uniform(BOUNDS[2][0], BOUNDS[2][1], N_TOTAL - 1),
        ]
    )
    points = np.vstack([np.asarray(GROUND_TRUTH, dtype=float), samples])
    frame = pd.DataFrame(points, columns=["thickness", "tilt_x", "tilt_y"])
    frame.insert(0, "point_index", np.arange(N_TOTAL))
    frame["is_ground_truth"] = False
    frame.loc[0, "is_ground_truth"] = True
    return frame


def simulate_scaled(thickness: float, tilt_x: float, tilt_y: float, apply_filter: bool) -> np.ndarray:
    device_simu = "gpu" if torch.cuda.is_available() else "cpu"
    image = simulate_cbed(
        thickness,
        tilt_x,
        tilt_y,
        PARAMS_ABTEM,
        device_simu=device_simu,
    )
    image = np.asarray(image, dtype=np.float64)
    if apply_filter:
        # Same 2D filter applied to each BO candidate (optimization.py, sigma=1).
        image = gaussian_filter(image, sigma=1)
    return image * OVERALL_SCALING_FACTOR


def patch_summaries(image: np.ndarray) -> torch.Tensor:
    reduction = ReductionFunction(
        {"reduction_type": "square", "reduction_kwargs": {"num_tiles": NUM_TILES}}
    )
    image_t = torch.tensor(image, dtype=torch.float64)
    patches = reduction(image_t)
    scale = torch.full((NUM_TILES ** 2,), SCALE_FACTOR, dtype=torch.float64)
    return patches.to(dtype=torch.float64) * scale


def pixel_and_patch_sse(image: np.ndarray, ground_truth: np.ndarray, true_patches: torch.Tensor):
    loss = LossFunction({"loss_type": "SSE", "dp_pow": 1})
    image_t = torch.tensor(image, dtype=torch.float64)
    truth_t = torch.tensor(ground_truth, dtype=torch.float64)
    pixel_sse = loss(image_t, truth_t, reduce=False)
    patches = patch_summaries(image)
    patch_sse = loss(patches, true_patches, reduce=False)
    pixel_sse = pixel_sse.reshape(1, 1)
    patch_sse = patch_sse.reshape(1, 1)
    epsilon, delta = solve(
        pixelSSE_val=pixel_sse,
        patchSSE_val=patch_sse,
        eps_base=EPS_BASE,
        eps_c=EPS_C,
    )
    return patches, float(epsilon.reshape(-1)[0]), float(delta.reshape(-1)[0]), float(patch_sse), float(pixel_sse)


def generate_ground_truth() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = sample_points()
    points.to_csv(POINTS_PATH, index=False)
    logger.info(f"Wrote {N_TOTAL} sample points to {POINTS_PATH}")

    # Ground truth is scaled the same way as in simulated_runner, without the candidate filter.
    image = simulate_scaled(*GROUND_TRUTH, apply_filter=False)
    imwrite(GROUND_TRUTH_PATH, image.astype(np.float32))
    logger.info(
        f"Saved ground truth image {GROUND_TRUTH_PATH} shape={image.shape} "
        f"min={image.min():.4f} max={image.max():.4f}"
    )


def evaluate_job(job_id: int) -> None:
    if not (0 <= job_id < N_JOBS):
        raise ValueError(f"job_id must be in 0..{N_JOBS - 1}, got {job_id}")
    if not GROUND_TRUTH_PATH.exists() or not POINTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {GROUND_TRUTH_PATH} or {POINTS_PATH}. Run --mode gt first."
        )

    print_system_info()
    from bott.io import load_tif

    ground_truth = np.asarray(load_tif(GROUND_TRUTH_PATH), dtype=np.float64)
    true_patches = patch_summaries(ground_truth)
    points = pd.read_csv(POINTS_PATH)
    start = job_id * N_PER_JOB
    chunk = points.iloc[start : start + N_PER_JOB]
    logger.info(f"Job {job_id}: evaluating point_index {start}..{start + N_PER_JOB - 1}")

    rows = []
    for row in chunk.itertuples(index=False):
        image = simulate_scaled(row.thickness, row.tilt_x, row.tilt_y, apply_filter=True)
        patches, epsilon, delta, patch_sse, pixel_sse = pixel_and_patch_sse(
            image, ground_truth, true_patches
        )
        record = {
            "point_index": int(row.point_index),
            "thickness": float(row.thickness),
            "tilt_x": float(row.tilt_x),
            "tilt_y": float(row.tilt_y),
            "is_ground_truth": bool(row.is_ground_truth),
            "epsilon": epsilon,
            "delta": delta,
            "patch_sse": patch_sse,
            "pixel_sse": pixel_sse,
        }
        patch_values = patches.detach().cpu().numpy().reshape(-1)
        record.update({name: float(value) for name, value in zip(PATCH_COLUMNS, patch_values)})
        true_values = true_patches.detach().cpu().numpy().reshape(-1)
        record.update({name: float(value) for name, value in zip(PATCH_TRUE_COLUMNS, true_values)})
        rows.append(record)
        logger.info(
            f"point {int(row.point_index)} "
            f"({row.thickness:.4f}, {row.tilt_x:.4f}, {row.tilt_y:.4f}) "
            f"pixel_sse={pixel_sse:.6g} patch_sse={patch_sse:.6g} "
            f"epsilon={epsilon:.6g} delta={delta:.6g}"
        )

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PARTS_DIR / f"job_{job_id:02d}.csv"
    columns = (
        ["point_index", "thickness", "tilt_x", "tilt_y", "is_ground_truth"]
        + PATCH_COLUMNS
        + ["epsilon", "delta", "patch_sse", "pixel_sse"]
        + PATCH_TRUE_COLUMNS
    )
    pd.DataFrame(rows)[columns].to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Grid-search evaluations for the simulated STO problem.")
    parser.add_argument("--mode", choices=["gt", "eval"], required=True)
    parser.add_argument("--job_id", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "eval" and args.job_id is None:
        parser.error("--job_id is required when --mode eval")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "gt":
        generate_ground_truth()
    else:
        evaluate_job(args.job_id)
