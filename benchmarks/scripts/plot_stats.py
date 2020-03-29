#!/usr/bin/env python

import argparse
import glob
import json

import numpy as np
import matplotlib.pyplot as plt


def main():
    data = json.load(open("results/fill_stats.json"))

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

    times = np.array(data["times"])
    fill_ratio = np.array(data["fill_ratio"])
    num_blocks = np.array(data["num_blocks"])
    capacity = np.array(data["capacity"])

    xs = (np.arange(len(times)) + 1) * 10

    axes[0].plot(xs, times, "-o", ms=0.5, alpha=0.5, label="elapsed time")
    axes[1].plot(xs, fill_ratio, "-o", ms=0.5, alpha=0.5, label="fill ratio")

    axes[2].plot(xs, capacity, "-o", ms=0.5, alpha=0.5, label="capacity")
    axes[2].plot(xs, num_blocks, "-o", ms=0.5, alpha=0.5, label="num blocks")
    axes[2].plot(xs, fill_ratio * capacity, "-o", ms=0.5, alpha=0.5, label="num avg leaf entries")

    for ax in axes:
        ax.legend(loc="best")

    plt.show()
    #import IPython; IPython.embed()


if __name__ == "__main__":
    main()

