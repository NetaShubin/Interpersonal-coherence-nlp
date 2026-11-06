import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_discourse_decay_curves(summary_csv_path="rollout_100conv_summary_results.csv"):
    if not os.path.exists(summary_csv_path):
        print(f"Error: Summary CSV not found at {summary_csv_path}. Please verify your file path.")
        return

    # Load data
    df = pd.read_csv(summary_csv_path)
    
    # Verify required columns exist
    required_cols = [
        "model_name", "turn_depth", "semantic_similarity", 
        "concession_markers", "justification_markers", 
        "lexical_overlap", "verbosity_ratio", "interrogative_density"
    ]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' missing from CSV.")
            return

    # Set aesthetic theme
    sns.set_theme(style="whitegrid", font="serif")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 15
    })

    # Define metrics and clean titles for plotting
    metrics = [
        ("semantic_similarity", "Semantic Similarity (Cosine)", [0.0, 1.0]),
        ("lexical_overlap", "Lexical Continuity (Overlap Ratio)", [0.0, 1.0]),
        ("verbosity_ratio", "Verbosity Ratio (Gen / Gold)", [0.0, 3.0]),
        ("interrogative_density", "Socratic / Interrogative Density", None),
        ("concession_markers", "Concession Marker Count", None),
        ("justification_markers", "Justification Marker Count", None)
    ]

    # Create a 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()

    # Color palette for models
    palette = sns.color_palette("deep", n_colors=df['model_name'].nunique())

    for idx, (metric_col, metric_title, y_lims) in enumerate(metrics):
        ax = axes[idx]
        
        # Plot lineplot with confidence intervals (ci=68 or errorbar='se')
        sns.lineplot(
            data=df,
            x="turn_depth",
            y=metric_col,
            hue="model_name",
            style="model_name",
            markers=True,
            dashes=False,
            errorbar="se", # standard error band across N=100 conversations
            palette=palette,
            linewidth=2,
            markersize=6,
            ax=ax
        )

        ax.set_title(metric_title, fontweight='bold', pad=10)
        ax.set_xlabel("Conversational Turn Depth ($t$)", labelpad=8)
        ax.set_ylabel("Metric Value", labelpad=8)
        
        if y_lims:
            ax.set_ylim(y_lims)

        # Clean up legend for individual subplots (keep on the first or last, or custom)
        if idx == 0:
            ax.legend(title="Model Architecture", frameon=True, framealpha=0.9, facecolor='white')
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    plt.suptitle("Multi-Turn Discourse Degradation & Decay Curves ($N=100$)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save output figure
    output_img = "carc_outputs/discourse_decay_curves.png"
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_discourse_decay_curves()
