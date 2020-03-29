#!/usr/bin/env python
# -*- encoding: utf-8

from __future__ import print_function, division, unicode_literals

import argparse
import glob
import json

from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Plot tool")
    """
    parser.add_argument(
        "files",
        nargs="+",
        help="JSON plot files"
    )
    """
    args = parser.parse_args()
    return args


def construct_color_map(keys):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = cycle(prop_cycle.by_key()['color'])

    colors = {}
    for key in keys:
        color = next(color_cycle)
        colors[key] = color

    return colors


def compute_stats(keys, data):
    stats = {}
    for key in keys:
        entries = [entry for entry in data if entry["name"] == key]
        final_values = np.array([entry["times"][-1] for entry in entries])
        stats[key] = final_values

    return stats


class ZBiasFreePlotter(object):
    def __init__(self):
        self.plot_calls = []

    def add_plot(self, f, xs, ys, *args, **kwargs):
        self.plot_calls.append((f, xs, ys, args, kwargs))

    def draw_plots(self, chunk_size=512):
        scheduled_calls = []
        for f, xs, ys, args, kwargs in self.plot_calls:
            assert(len(xs) == len(ys))
            index = np.arange(len(xs))
            np.random.shuffle(index)
            index_blocks = [index[i:i+chunk_size] for i in np.arange(len(index))[::chunk_size]]
            for i, index_block in enumerate(index_blocks):
                # Only attach a label for one of the chunks
                if i != 0 and kwargs.get("label") is not None:
                    kwargs = kwargs.copy()
                    kwargs["label"] = None
                scheduled_calls.append((f, xs[index_block], ys[index_block], args, kwargs))

        np.random.shuffle(scheduled_calls)

        for f, xs, ys, args, kwargs in scheduled_calls:
            f(xs, ys, *args, **kwargs)


def main():
    args = parse_args()

    files = glob.glob("results/fill_avg_*.json")

    data = [
        json.load(open(f))
        for f in sorted(files)
    ]

    keys = [entry["name"] for entry in data if entry["run"] == 1]

    color_map = construct_color_map(keys)
    stats = compute_stats(keys, data)

    # import IPython; IPython.embed()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    bias_free_plotter1 = ZBiasFreePlotter()
    bias_free_plotter2 = ZBiasFreePlotter()

    fig.text(0.78, 0.93, "Total times [ms]", fontsize=9, family="monospace", weight="bold")
    y_text = 0.9

    for i, entry in enumerate(data):
        name = entry["name"]
        iters = np.array(entry["iters"])
        times = np.array(entry["times"]) * 1000

        color = color_map[name]
        is_primary = entry["run"] == 1

        if is_primary:
            label = name

            mean = stats[name].mean() * 1000
            std = stats[name].std() * 1000
            fig.text(0.77, y_text, name, fontsize=9, family="monospace")
            fig.text(0.87, y_text, "{:7.3f}".format(mean), fontsize=9, family="monospace")
            fig.text(0.93, y_text, "± {:6.3f}".format(std), fontsize=9, family="monospace")
            y_text -= 0.03
        else:
            label = None

        axes[0].plot(
            iters, times, "-",
            c=color, alpha=0.5, label=label,
        )
        if False:
            axes[1].plot(
                iters[1:], times[1:] - times[:-1],
                "o", c=color, ms=0.4, alpha=0.8, label=label,
            )
        else:
            bias_free_plotter1.add_plot(
                axes[1].plot, iters[1:], times[1:] - times[:-1],
                ",", c=color, ms=1, alpha=1, label=label
            )
            bias_free_plotter2.add_plot(
                axes[1].plot, iters[1:], times[1:] - times[:-1],
                "o", c=color, ms=4, alpha=0.007,
            )

    bias_free_plotter1.draw_plots()
    bias_free_plotter2.draw_plots()

    axes[0].legend(loc="best", prop={'size': 9})
    axes[0].grid(color="#DDDDDD")
    axes[1].grid(color="#DDDDDD")

    axes[0].set_title("Total time elapsed", fontsize=10)
    axes[1].set_title("Delta times", fontsize=10)

    axes[0].set_ylabel("Time [ms]")
    axes[1].set_ylabel("Time [ms]")
    axes[1].set_xlabel("Operations")
    axes[1].set_yscale("log")

    fig.tight_layout()
    plt.subplots_adjust(right=0.75)

    plt.show()


if __name__ == "__main__":
    main()

