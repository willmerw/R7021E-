"""Run logging and the metrics the report is built from."""
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np


PACKAGE = 'r7021e_fast_slam'


def package_root() -> Path:
    """The package's SOURCE directory, wherever the node was launched from.

    A relative `log_dir` has to mean the same place every time. `ros2 launch`
    runs the node with your shell's working directory, so a bare "runs" would
    otherwise scatter run logs across every directory you happened to be in.

    colcon installs a standalone copy of the Python package, with no link back
    to the source tree, so the source directory has to be found rather than
    derived from `__file__`. Three steps, first hit wins:

      1. `__file__` is already inside a directory tree containing a
         `package.xml` -- running from the source checkout, or from the tests.
      2. The ament prefix names the workspace (`<ws>/install/<pkg>`), so walk
         up to the directory holding both `install` and `src` and look for the
         package underneath `src`.
      3. Neither worked: fall back to `~/.ros/<package>`, which is at least
         always the same place.
    """
    for d in Path(__file__).resolve().parents:
        if (d / 'package.xml').is_file():
            return d

    for prefix in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep):
        if not prefix:
            continue
        for ws in Path(prefix).resolve().parents:
            src = ws / 'src'
            if not src.is_dir():
                continue
            ## The package may sit at any depth under src/.
            for manifest in src.glob(f'**/{PACKAGE}/package.xml'):
                return manifest.parent

    fallback = Path.home() / '.ros' / PACKAGE
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def resolve_log_dir(log_dir) -> Path:
    """Absolute `log_dir` is used as given; relative is under the package."""
    log_dir = Path(log_dir)
    return log_dir if log_dir.is_absolute() else package_root() / log_dir


@dataclass
class StepInfo:
    """What one filter update reports about itself.

    Lives here rather than in `rbpf.py` because the run log, the figures and
    the mapping-only path in the node all need it, and none of them should
    depend on a student module that may not exist yet.
    """
    n_eff: float          # pre-resample: the value that made the decision
    resampled: bool = False
    n_fallback: int = 0
    best_index: int = 0
    timings: Dict[str, float] = field(
        default_factory=lambda: {'scan_match': 0.0, 'likelihood': 0.0,
                                 'map_integrate': 0.0})


class RunLogger:
    """Accumulates one record per filter step and writes a single .npz."""

    def __init__(self, cfg):
        self.dir = resolve_log_dir(cfg.log_dir)
        self.name = cfg.run_name
        self.params = asdict(cfg)
        self._rows = []
        self._final_map = None

    def add(self, t, est_pose, odom_pose, gt_pose, info, map_bytes):
        nan3 = np.full(3, np.nan)
        self._rows.append((
            float(t),
            np.asarray(est_pose, dtype=float),
            np.asarray(odom_pose, dtype=float),
            nan3 if gt_pose is None else np.asarray(gt_pose, dtype=float),
            float(info.n_eff), bool(info.resampled), int(info.n_fallback),
            float(info.timings['scan_match']),
            float(info.timings['likelihood']),
            float(info.timings['map_integrate']),
            int(info.best_index), int(map_bytes),
        ))

    def set_final_map(self, grid):
        """Snapshot the best particle's grid; the map figures need it."""
        self._final_map = (grid.log_odds.copy(), float(grid.res),
                           np.asarray(grid.origin, dtype=float).copy())

    def save(self):
        """Write <log_dir>/<run_name>.npz plus a .json sidecar of the params."""
        if not self._rows:
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        cols = list(zip(*self._rows))
        gt = np.stack(cols[3])
        path = self.dir / f'{self.name}.npz'

        if self._final_map is None:
            map_kw = dict(has_map=np.array(False),
                          map_log_odds=np.zeros((1, 1), dtype=np.float32),
                          map_resolution=np.array(0.0),
                          map_origin=np.zeros(2))
        else:
            lo, res, origin = self._final_map
            map_kw = dict(has_map=np.array(True), map_log_odds=lo,
                          map_resolution=np.array(res), map_origin=origin)

        np.savez_compressed(
            path,
            t=np.array(cols[0]),
            est=np.stack(cols[1]),
            odom=np.stack(cols[2]),
            gt=gt,
            n_eff=np.array(cols[4]),
            resampled=np.array(cols[5]),
            n_fallback=np.array(cols[6]),
            t_scan_match=np.array(cols[7]),
            t_likelihood=np.array(cols[8]),
            t_map_integrate=np.array(cols[9]),
            best_index=np.array(cols[10]),
            map_bytes=np.array(cols[11]),
            has_ground_truth=np.array(bool(np.isfinite(gt).all())),
            **map_kw,
        )
        path.with_suffix('.json').write_text(json.dumps(self.params, indent=2,
                                                        default=str))
        return path


def load_run(path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        run = {k: data[k] for k in data.files}
    run['has_ground_truth'] = bool(run['has_ground_truth'])
    run['has_map'] = bool(run['has_map'])
    sidecar = path.with_suffix('.json')
    run['params'] = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return run


def _require_ground_truth(gt):
    gt = np.asarray(gt, dtype=float)
    if not np.isfinite(gt).all():
        raise ValueError(
            'no ground truth in this run; use the surrogate metrics instead')
    return gt


def absolute_trajectory_error(est, gt) -> dict:
    """Position ATE. No alignment: both trajectories start at the
    origin, so aligning would hide a genuine initialisation error."""
    gt = _require_ground_truth(gt)
    est = np.asarray(est, dtype=float)
    err = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
    return {'rmse': float(np.sqrt(np.mean(err ** 2))),
            'mean': float(err.mean()),
            'max': float(err.max()),
            'per_step': err}

