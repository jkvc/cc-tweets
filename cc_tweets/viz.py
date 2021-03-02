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
