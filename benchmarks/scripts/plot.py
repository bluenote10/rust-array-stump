#!/usr/bin/env python
# -*- encoding: utf-8

from __future__ import print_function, division, unicode_literals

import argparse
import glob
import json

from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Ubuntu']})
rc('font', **{'monospace': ['Ubuntu Mono']})

rc('axes', titlesize="11")
rc('axes', labelsize="9")
rc('xtick', labelsize="9")
rc('ytick', labelsize="9")


def parse_args():
    parser = argparse.ArgumentParser("Plot tool")
    parser.add_argument(
        "-b", "--bench-mode",
        help="The 'benchmark mode' part of the file prefix"
    )
    parser.add_argument(
        "-g", "--gen-mode",
        help="The 'generator mode' part of the file prefix"
    )
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


def set_share_axes(ax, ax_target):
    # https://stackoverflow.com/a/51684195/1804173
    ax_target._shared_x_axes.join(ax_target, ax)
    ax_target.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
    ax_target.xaxis.offsetText.set_visible(False)


def main():
    args = parse_args()
    bench_mode = args.bench_mode
    gen_mode = args.gen_mode

    files = glob.glob("results/{}_{}_*.json".format(bench_mode, gen_mode))

    if False:
        # Hack to display "within group" comparisons. Should we fully support that?
        files = glob.glob("results/insert_*_ArrayStump_*.json".format(bench_mode, gen_mode))

        def patch(data, fn):
            if "avg" in fn:
                data["name"] = data["name"] + "AVG"
            if "asc" in fn:
                data["name"] = data["name"] + "ASC"
            if "dsc" in fn:
                data["name"] = data["name"] + "DSC"
            return data

        data = [
            patch(json.load(open(f)), f)
            for f in sorted(files)
        ]

    data = [
        json.load(open(f))
        for f in sorted(files)
    ]

    keys = [entry["name"] for entry in data if entry["run"] == 1]

    color_map = construct_color_map(keys)
    stats = compute_stats(keys, data)

    # import IPython; IPython.embed()

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.5))
    set_share_axes(axes[1], axes[0])

    bias_free_plotter1 = ZBiasFreePlotter()
    bias_free_plotter2 = ZBiasFreePlotter()

    line_spacing = 0.024
    y_text = 0.91
    fig.text(0.77, y_text + line_spacing, "Total elapsed times [ms]", fontsize=9, weight="bold")

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
            fig.text(0.77, y_text, name, fontsize=9)
            fig.text(0.91, y_text, "{:.3f}".format(mean), fontsize=9, ha="right")
            fig.text(0.97, y_text, "± {:6.3f}".format(std), fontsize=9, ha="right")
            y_text -= line_spacing
        else:
            label = None

        deltas_xs = iters[1:]
        deltas_ys = (times[1:] - times[:-1]) / entry["measure_every"]

        axes[0].plot(
            iters, times, "-",
            c=color, alpha=0.5, label=label,
        )
        if False:
            axes[1].plot(
                deltas_xs, deltas_ys,
                "o", c=color, ms=0.4, alpha=0.8, label=label,
            )
        else:
            for ax in axes[1:]:
                bias_free_plotter1.add_plot(
                    ax.plot, deltas_xs, deltas_ys,
                    ",", c=color, ms=1, alpha=1, label=label
                )
                bias_free_plotter2.add_plot(
                    ax.plot, deltas_xs, deltas_ys,
                    "o", c=color, ms=4, alpha=0.007,
                )

    bias_free_plotter1.draw_plots()
    bias_free_plotter2.draw_plots()

    axes[0].legend(loc="best", prop={'size': 9})
    for ax in axes:
        ax.grid(color="#DDDDDD")
        ax.set_facecolor('#FCFEFF')

    axes[0].set_title("Total time elapsed", fontsize=10)
    axes[1].set_title("Delta times (semi-log)", fontsize=10)
    axes[2].set_title("Delta times (log-log)", fontsize=10)

    axes[0].set_ylabel("Time [ms]")
    axes[1].set_ylabel("Time / op [ms]")
    axes[1].set_xlabel("Operations")
    axes[2].set_ylabel("Time / op [ms]")
    axes[2].set_xlabel("Operations")

    axes[1].set_yscale("log")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")

    fig.tight_layout()
    plt.subplots_adjust(right=0.75)

    plt.savefig("results/{}_{}_comparison.png".format(bench_mode, gen_mode))
    plt.show()


if __name__ == "__main__":
    main()

