import argparse
import os
import json
import glob
import re
import matplotlib.pyplot as plt
import numpy as np


def get_display_name(raw_name):
    """
    Map raw directory name to display name based on MODEL_MAPPING keys.
    """
    raw_lower = raw_name.lower()
    # Priority matching: check for explicit prefixes/substrings
    # We use specific logic based on the known directory structure
    if raw_lower.startswith("base"):
        return "$R_{base}$"
    if raw_lower.startswith("acc"):
        return "$R_{base} + R_{rep} + R_{beh}$"
    # 'softall' could be substring of others, but 'softallfull' is specific
    if "softall" in raw_lower and "acc" not in raw_lower:
        return "$R_{base} + R_{rep}$"
    # if "softmid" in raw_lower: return "$R_{softmid}$"
    # if "softlast2" in raw_lower: return "$R_{softlast2}$"

    return raw_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate Layer-wise Similarity Curves with 95% CI"
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="./alignment_analysis",
        help="Root directory containing all experimental results. Structure should be: root_dir/dataset_name/model_name/layer_stats.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./aggregated_plots",
        help="Folder to save generated plots",
    )
    parser.add_argument(
        "--filename_pattern",
        type=str,
        default="layer_stats.json",
        help="JSON filename to look for",
    )

    return parser.parse_args()


def extract_metadata_from_path(filepath, root_dir):
    rel_path = os.path.relpath(filepath, root_dir)
    parts = rel_path.split(os.sep)

    if len(parts) >= 3:
        dataset_name = parts[-3]
        model_name = parts[-2]
    elif len(parts) == 2:
        dataset_name = "Default"
        model_name = parts[-2]
    else:
        dataset_name = "Unknown"
        model_name = "Unknown"

    return dataset_name, model_name


def load_data(root_dir, pattern):
    """
    Load data, including means and confidence intervals.
    Returns: {dataset: {model: {'layers': [], 'means': [], 'ci_low': [], 'ci_high': []}}}
    """
    search_path = os.path.join(root_dir, "**", pattern)
    files = glob.glob(search_path, recursive=True)

    data_tree = {}

    print(f"Found {len(files)} result files.")

    for f in files:
        dataset, model = extract_metadata_from_path(f, root_dir)

        try:
            with open(f, "r") as fp:
                content = json.load(fp)

            layers, means, ci_low, ci_high = [], [], [], []

            if "layers" in content and "means" in content:
                layers = content["layers"]
                means = content["means"]
                ci_low = content.get("ci_lower", [])
                ci_high = content.get("ci_upper", [])
            else:
                print(
                    f"Skipping {f}: Format not supported (missing 'layers' or 'means')"
                )
                continue

            if len(layers) != len(means):
                print(f"Warning: Length mismatch in {f}")
                continue

            if dataset not in data_tree:
                data_tree[dataset] = {}

            layers = [l + 1 for l in layers]  # update the index.

            data_tree[dataset][model] = {
                "layers": layers,
                "means": means,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
            print(f"Loaded: {dataset} | {model} (Layers: {len(layers)})")

        except Exception as e:
            print(f"Error reading {f}: {e}")

    return data_tree


def plot_aggregated_similarity(data_tree, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # =========================================================
    # NEW CODE: Side-by-Side Plotting for Paper
    # =========================================================

    # 1. Configuration
    # Define models and their fixed styles to ensure consistency across subplots
    model_styles = {
        "$R_{base}$": {"color": "#1f77b4", "marker": "o"},  # Blue
        "$R_{base} + R_{rep}$": {"color": "#ff7f0e", "marker": "o"},  # Orange
        "$R_{base} + R_{rep} + R_{beh}$": {"color": "#2ca02c", "marker": "o"},  # Green
    }

    target_models = list(model_styles.keys())

    # 2. Identify Datasets from data_tree
    # Assuming keys contain "MMSU" and "OBQA" substrings
    mmsu_key = next((k for k in data_tree.keys() if "MMSU" in k), None)
    obqa_key = next((k for k in data_tree.keys() if "OBQA" in k), None)

    if not mmsu_key or not obqa_key:
        print(
            f"Error: Could not find both MMSU and OBQA in data_tree keys: {list(data_tree.keys())}"
        )
        return

    # 3. Create Canvas (Side-by-Side)
    # sharey=True hides the right Y-axis ticks automatically
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
    plt.subplots_adjust(wspace=0.02, bottom=0.2)  # Adjust spacing

    # 4. Helper Function to Plot on a specific Axis
    def plot_subplot(ax, dataset_key, subplot_label):
        models_data = data_tree[dataset_key]

        # Mapping logic (from your original code)
        mapped_models = {}
        for raw_model_name, m_data in models_data.items():
            display_name = get_display_name(raw_model_name)
            mapped_models[display_name] = m_data

        # Iterate through the fixed order of target_models
        max_layer = 0
        for model_name in target_models:
            if model_name not in mapped_models:
                continue

            m_data = mapped_models[model_name]
            layers = m_data["layers"]
            means = m_data["means"]
            ci_low = m_data["ci_low"]
            ci_high = m_data["ci_high"]

            max_layer = max(max_layer, max(layers))
            style = model_styles[model_name]

            # Plot Line & Marker
            ax.plot(
                layers,
                means,
                label=model_name,
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=2,
                markersize=3.5,
                alpha=0.9,
            )

            # Plot CI
            if len(ci_low) == len(layers) and len(ci_high) == len(layers):
                ax.fill_between(
                    layers,
                    ci_low,
                    ci_high,
                    color=style["color"],
                    alpha=0.4,
                    linewidth=0,
                )

        # Axis Styling
        ax.set_ylim(0.92, 1.0)
        ax.set_xlim(0.5, max_layer + 0.5)
        ax.set_xticks(np.arange(1, max_layer + 1, step=2))
        ax.grid(True, which="major", linestyle="--", color="grey", alpha=0.25)

        # Subplot Label (e.g., "(a) MMSU Dataset") placed at the bottom
        ax.set_title(subplot_label, y=-0.25, fontsize=12, fontweight="normal")

    # 5. Execute Plotting
    plot_subplot(ax1, mmsu_key, "(a) MMSU Dataset")
    plot_subplot(ax2, obqa_key, "(b) OBQA Dataset")

    # 6. Global Labels & Legend

    # Y-Axis Label (Only on the left plot)
    ax1.set_ylabel("Cosine Similarity", fontsize=12)
    ax2.set_ylabel("")  # Ensure right plot has no label

    # X-Axis Label (Centered for the whole figure)
    # y=0.02 puts it right above the figure bottom edge
    fig.supxlabel("Layer Depth", fontsize=12, y=0.085)

    # Legend (Shared, placed at top center)
    handles, labels = ax1.get_legend_handles_labels()
    # Ensure legend order matches target_models order
    # (Since we iterated target_models, it should already be correct,
    # but dicts in python 3.7+ preserve insertion order, so this is safe)
    fig.legend(handles, labels, loc="upper center", ncol=3, prop={"size": 12})

    # 7. Save
    output_filename = "layer_sim_analysis_combined.pdf"
    save_path = os.path.join(output_dir, output_filename)

    # Use bbox_inches='tight' to ensure legend isn't cropped
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Combined plot saved to: {save_path}")
    plt.close()


def main():
    args = parse_args()

    print(f"Scanning root: {args.root_dir}...")
    data_tree = load_data(args.root_dir, args.filename_pattern)

    if not data_tree:
        print("No data found.")
        return

    plot_aggregated_similarity(data_tree, args.output_dir)


if __name__ == "__main__":
    main()
