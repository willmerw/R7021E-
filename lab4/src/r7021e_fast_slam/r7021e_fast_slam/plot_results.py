"""Figures for the lab report, built from the .npz run logs.

Every figure is a function taking loaded run dicts and an output path, so the
same code serves the CLI and the tests.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

from .utils import expit
from .evaluation import absolute_trajectory_error, load_run

STAGES = ('t_scan_match', 't_likelihood', 't_map_integrate')
STAGE_LABELS = ('scan match', 'likelihood', 'map integrate')


def _plt():
    """Import pyplot late and force a headless backend for batch runs."""
    if matplotlib.get_backend().lower() not in ('agg', 'module://agg'):
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    _plt().close(fig)
    return out


def _map_image(run):
    if not run['has_map']:
        raise ValueError('this run has no final map snapshot')
    lo = run['map_log_odds']
    img = expit(lo)                    
    img[lo == 0.0] = np.nan          # unobserved cells stay blank
    return img


def _extent(run):
    lo = run['map_log_odds']
    res = float(run['map_resolution'])
    ox, oy = run['map_origin']
    return [ox, ox + lo.shape[1] * res, oy, oy + lo.shape[0] * res]


def plot_maps(runs, labels, out):
    """Handout figure 1: maps side by side, with each run's path on top."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5),
                             squeeze=False)
    for ax, run, label in zip(axes[0], runs, labels):
        ax.imshow(_map_image(run), origin='lower', extent=_extent(run),
                  cmap='Greys', vmin=0.0, vmax=1.0)
        ax.plot(run['est'][:, 0], run['est'][:, 1], lw=1.5, color='tab:green',
                label='estimate')
        if run['has_ground_truth']:
            ax.plot(run['gt'][:, 0], run['gt'][:, 1], lw=1.0, ls='--',
                    color='tab:blue', label='ground truth')
        ax.set_title(label)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=8)
    return _save(fig, out)


def plot_ate_vs_particles(runs, out):
    """Handout figure 2: ATE against N, one series
    per proposal, spread taken over seeds."""
    plt = _plt()
    groups = {}
    for run in runs:
        if not run['has_ground_truth']:
            raise ValueError('ATE needs ground truth')
        key = (bool(run['params'].get('use_improved_proposal', True)),
               int(run['params'].get('num_particles', 1)))
        ate = absolute_trajectory_error(run['est'], run['gt'])['rmse']
        groups.setdefault(key, []).append(ate)

    fig, ax = plt.subplots(figsize=(6, 4))
    for improved, name, colour in ((True, 'Grid-FastSLAM 2.0', 'tab:green'),
                                   (False, 'FastSLAM 1.0 (odometry)',
                                    'tab:red')):
        ns = sorted(n for (imp, n) in groups if imp == improved)
        if not ns:
            continue
        means = [np.mean(groups[(improved, n)]) for n in ns]
        lo = [np.min(groups[(improved, n)]) for n in ns]
        hi = [np.max(groups[(improved, n)]) for n in ns]
        ax.plot(ns, means, 'o-', color=colour, label=name)
        ax.fill_between(ns, lo, hi, color=colour, alpha=0.2)
    ax.set_xscale('log')
    ax.set_xlabel('number of particles N')
    ax.set_ylabel('ATE RMSE [m]')
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, out)


def plot_neff(runs, labels, out):
    """Handout figure 3: N_eff over time for the two resampling policies.

    Plotted as the fraction N_eff/N so runs with different particle counts
    share one axis, which is also the unit `resample_threshold` is given in.
    The first step is dropped: the particles start with equal weights, so it
    is always exactly 1.0 and it flattens everything that follows.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for run, label in zip(runs, labels):
        n = int(run['params'].get('num_particles', 1))
        t = (run['t'] - run['t'][0])[1:]
        frac = run['n_eff'][1:] / n
        events = run['resampled'].astype(bool)[1:]
        line, = ax.plot(t, frac, lw=1.2,
                        label=f'{label} (N={n}, resampled '
                              f'{events.mean() * 100:.0f}% of steps, '
                              f'median {np.median(frac):.2f})')
        ## Rug inside the bottom of the axes rather than markers on the
        ## curve: with a near-100% resample rate, on-curve markers bury the curve itself.
        rug_y = 0.02 + 0.025 * (len(ax.lines) - 1)
        ax.plot(t[events], np.full(events.sum(), rug_y), '|', ms=5, alpha=0.7,
                color=line.get_color())

    for thr in sorted({float(r['params'].get('resample_threshold', np.nan))
                       for r in runs}):
        if np.isfinite(thr):
            ax.axhline(thr, ls='--', lw=1.0, color='0.4')
            ax.text(1.0, thr, f' threshold {thr:g}', va='bottom', ha='right',
                    color='0.4', fontsize=8, transform=ax.get_yaxis_transform())

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('time [s]')
    ax.set_ylabel(r'$N_{\mathrm{eff}}\,/\,N$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out)


def plot_timing(runs, out):
    """Handout figure 4: per-update time split by stage, and map memory."""
    plt = _plt()
    order = sorted(runs, key=lambda r: int(r['params'].get('num_particles', 1)))
    ns = [int(r['params'].get('num_particles', 1)) for r in order]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(order))
    for stage, label in zip(STAGES, STAGE_LABELS):
        vals = np.array([float(np.mean(r[stage])) * 1e3 for r in order])
        ax.bar(x, vals, 0.6, bottom=bottom, label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel('number of particles N')
    ax.set_ylabel('mean time per update [ms]')
    ax.legend(loc='upper left')

    mem = ax.twinx()
    mem.plot(x, [float(np.mean(r['map_bytes'])) / 1e6 for r in order],
             'k^--', label='map memory')
    mem.set_ylabel('total map memory [MB]')
    return _save(fig, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Figures from Grid-FastSLAM runs.')
    ap.add_argument('--runs', nargs='+', required=True, help='.npz run logs')
    ap.add_argument('--out', default='figures', help='output directory')
    ap.add_argument('--figure', required=True,
                    choices=['maps', 'ate', 'neff', 'timing'])
    ap.add_argument('--labels', nargs='*', default=None)
    args = ap.parse_args(argv)

    runs = [load_run(p) for p in args.runs]
    out_dir = Path(args.out)
    labels = args.labels or [Path(p).stem for p in args.runs]

    if args.figure == 'maps':
        plot_maps(runs, labels, out_dir / 'maps.png')
    elif args.figure == 'ate':
        plot_ate_vs_particles(runs, out_dir / 'ate_vs_particles.png')
    elif args.figure == 'neff':
        plot_neff(runs, labels, out_dir / 'neff.png')
    elif args.figure == 'timing':
        plot_timing(runs, out_dir / 'timing.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
