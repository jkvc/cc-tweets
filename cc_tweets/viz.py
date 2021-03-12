import matplotlib.pyplot as plt
import numpy as np


def grouped_bars(fig, ax, x_labels, name2series, width=0.35):
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    x = np.arange(len(x_labels))  # the label locations
    # fig, ax = plt.subplots()
    for i, (name, series) in enumerate(name2series.items()):
        rects = ax.bar(x + width * i, series, width, label=name)
        autolabel(rects)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    return fig, ax


def plot_feature_weights(name2weight, save_path, title, xlim=None):
    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 7))
    y_pos = np.arange(len(name2weight))
    ax.barh(
        y_pos,
        name2weight.values(),
        align="center",
    )
    ax.yaxis.tick_right()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"[{round(weight, 4)}] {name}" for name, weight in name2weight.items()],
    )
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    plt.subplots_adjust(left=0.1, right=0.6)
    plt.savefig(save_path)
    plt.clf()
