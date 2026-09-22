#!/usr/bin/env bash
## Replay ONE bag through the filter once per (proposal, N, seed) combination.
##
## The central experiment of the lab is 2 proposals x 4 particle counts x 3
## seeds = 24 runs. Driving the maze 24 times would take the whole day and
## would compare 24 different trajectories, so it would not be a controlled
## experiment at all. Record one good drive, then replay that same bag here:
## every run sees identical (u, z) input and the only thing that changes is
## what you set out to change.
##
##   tools/run_sweep.sh runs/maze_drive
##
## Writes runs/<name>.npz per run, which is what plot_results.py reads:
##
##   python3 -m r7021e_fast_slam.plot_results --figure ate --runs runs/*.npz

set -euo pipefail
## Job control. Without it this script only works when run from an interactive
## terminal: a non-interactive shell sets SIGINT to ignore on every background
## job, Python keeps an inherited SIG_IGN, and the `kill -INT` below is then a
## silent no-op that leaves the node running and `wait` blocked forever.
set -m

BAG=${1:?usage: run_sweep.sh <bag_dir> [particle_counts] [seeds]}
NS=${2:-"1 5 10 15 20 25 50"}
SEEDS=${3:-"1 2 3"}

for improved in true false; do
  for n in $NS; do
    for s in $SEEDS; do
      tag="$([ "$improved" = true ] && echo fs2 || echo fs1)_n${n}_s${s}"
      echo "=== $tag"
      ros2 launch r7021e_fast_slam r7021e_fast_slam.launch.py \
        use_rviz:=false use_sim_time:=true \
        num_particles:="$n" seed:="$s" \
        use_improved_proposal:="$improved" run_name:="$tag" &
      launch_pid=$!

      ## Give the node time to come up before the bag starts, or the first
      ## scans arrive with nothing subscribed and the run is short a few steps.
      sleep 5
      ros2 bag play "$BAG" --clock

      ## SIGINT, not SIGKILL: the node saves its run log on shutdown.
      kill -INT $launch_pid
      wait $launch_pid 2>/dev/null || true
    done
  done
done
echo "done -- run logs are in runs/"
