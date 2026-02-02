#!/usr/bin/env python3
import argparse
import pathlib

import matplotlib.pyplot as plt


def load_trajectory(path: pathlib.Path):
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                t = float(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError:
                continue
            samples.append((t, x, y, z))
    samples.sort(key=lambda item: item[0])
    ts, xs, ys, zs = [], [], [], []
    for t, x, y, z in samples:
        ts.append(t)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return ts, xs, ys, zs


def main():
    parser = argparse.ArgumentParser(description="Plot ORB-SLAM3 trajectory.")
    parser.add_argument("trajectory", type=pathlib.Path, help="Path to trajectory txt")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="Output image path")
    parser.add_argument("--show", action="store_true", help="Show interactive plot")
    parser.add_argument("--time-series", action="store_true", help="Plot x/y/z vs time")
    args = parser.parse_args()

    if not args.trajectory.exists():
        raise SystemExit(f"Trajectory not found: {args.trajectory}")

    ts, xs, ys, zs = load_trajectory(args.trajectory)
    if not xs:
        raise SystemExit("No trajectory points parsed.")

    if args.time_series:
        t0 = ts[0]
        rel_t = [t - t0 for t in ts]
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(rel_t, xs, linewidth=1.2)
        axes[0].set_ylabel("x")
        axes[1].plot(rel_t, ys, linewidth=1.2)
        axes[1].set_ylabel("y")
        axes[2].plot(rel_t, zs, linewidth=1.2)
        axes[2].set_ylabel("z")
        axes[2].set_xlabel("time (s)")
        fig.suptitle(args.trajectory.name)
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, linewidth=1.5)
        ax.scatter(xs[0], ys[0], zs[0], c="green", s=20, label="start")
        ax.scatter(xs[-1], ys[-1], zs[-1], c="red", s=20, label="end")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(args.trajectory.name)
        ax.legend()
        ax.view_init(elev=25, azim=-60)
        fig.tight_layout()

    out_path = args.out
    if out_path is None:
        out_path = args.trajectory.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

"""
python3 /home/yanwq/project003/ORB_SLAM3/Examples/MultiFisheye/plot_trajectory.py \
/home/yanwq/project003/ORB_SLAM3/Examples/MultiFisheye/output2_rig_dir/trajectory_cam0.txt \
--show
"""
