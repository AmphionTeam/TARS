import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12


def draw_chart():
    categories = ["Shallow", "Middle", "Deep", "Last", "All"]

    # Audio Scores (Avg. A)
    audio_scores = [72.62, 75.48, 73.31, 72.34, 75.83]

    # Text Scores (Avg. T - Reference) - REMOVED

    x = np.arange(len(categories))
    width = 0.5

    fig, ax = plt.subplots(figsize=(5, 2.5))

    rects1 = ax.bar(
        x,
        audio_scores,
        width,
        label="Audio (Avg. A)",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

    ax.set_ylabel("Average Accuracy (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylim(70, 78)

    ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)
    ax.set_axisbelow(True)

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, frameon=False, prop={'weight': 'bold', 'size': 12})

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    autolabel(rects1)

    plt.tight_layout()

    output_path = "layer_sensitivity.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Chart saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    draw_chart()
