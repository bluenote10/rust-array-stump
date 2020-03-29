#!/usr/bin/env python

import json

import numpy as np
import matplotlib.pyplot as plt


def filter_nones(values):
    return np.array([v if v is not None else np.nan for v in values])


def main():
    data = json.load(open("results/fill_stats.json"))

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

    times = filter_nones(data["times"])
    fill_ratio = filter_nones(data["fill_ratio"])
    fill_min = filter_nones(data["fill_min"])
    fill_max = filter_nones(data["fill_max"])
    num_blocks = filter_nones(data["num_blocks"])
    capacity = filter_nones(data["capacity"])

    xs = (np.arange(len(times)) + 1) * 10

    axes[0].plot(xs, times, "-o", ms=0.5, alpha=0.5, label="elapsed time")
    axes[1].plot(xs, fill_ratio, "-o", ms=0.5, alpha=0.5, label="fill ratio")

    axes[2].plot(xs, capacity, "-o", ms=0.5, alpha=0.5, label="capacity")
    axes[2].plot(xs, num_blocks, "-o", ms=0.5, alpha=0.5, label="num blocks")

    axes[2].plot(xs, fill_min, "-o", ms=0.5, alpha=0.5, label="num min leaf entries")
    axes[2].plot(xs, fill_ratio * capacity, "-o", ms=0.5, alpha=0.5, label="num avg leaf entries")
    axes[2].plot(xs, fill_max, "-o", ms=0.5, alpha=0.5, label="num max leaf entries")

    for ax in axes:
        ax.legend(loc="best")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

