import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_deconfounding_heatmap(json_path: str, task: str = "esi", format_type: str = "EXP"):
    """
    Generates a side-by-side heatmap comparing Naive Accuracy Disparity (AD) 
    vs. Paraphrase-Corrected Excess Mean.
    """
    path = Path(json_path)
    if not path.exists():
        print(f"File not found: {json_path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    demo_data = data.get("demographic_bias", {})
    models = list(demo_data.keys())
    attributes = ["ses", "race", "sex"]
    
    # Initialize data structures for the heatmaps
    ad_matrix = np.zeros((len(attributes), len(models)))
    pe_matrix = np.zeros((len(attributes), len(models)))
    
    for i, attr in enumerate(attributes):
        for j, model in enumerate(models):
            # Extract Naive AD
            try:
                ad = demo_data[model][attr][format_type][task]["ad"]["ad"]
            except KeyError:
                ad = np.nan
            ad_matrix[i, j] = ad
            
            # Extract Paraphrase Excess Mean
            try:
                pe = demo_data[model][attr]["paraphrase_excess"][format_type][task]["mean"]
            except KeyError:
                pe = np.nan
            pe_matrix[i, j] = pe

    # Create DataFrames for Seaborn
    df_ad = pd.DataFrame(ad_matrix, index=[a.upper() for a in attributes], columns=models)
    df_pe = pd.DataFrame(pe_matrix, index=[a.upper() for a in attributes], columns=models)

    # Determine shared color scale (vmin, vmax) to ensure 1:1 visual comparison
    # AD is strictly positive, PE can be negative. We center at 0.
    max_val = max(np.nanmax(np.abs(ad_matrix)), np.nanmax(np.abs(pe_matrix)))
    # Add a little headroom to the max value for color scaling
    vmax = round(max_val + 0.02, 2)
    vmin = -vmax

    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f"Demographic Bias in {task.upper()} ({format_type} Formatting): Naive vs. De-Confounded", 
                 fontsize=16, y=1.05, fontweight='bold')

    # Custom Diverging Colormap: Red for positive shift (bias), Blue for negative, White for zero
    cmap = sns.color_palette("vlag", as_cmap=True)

    # 1. Left Heatmap: Naive Benchmark (Illusion)
    sns.heatmap(df_ad, ax=axes[0], cmap=cmap, center=0, vmin=vmin, vmax=vmax, 
                annot=True, fmt=".3f", cbar=False, linewidths=.5, square=True,
                annot_kws={"size": 12, "weight": "bold"})
    axes[0].set_title("Naive Benchmark (Accuracy Disparity)", fontsize=14, pad=15)
    axes[0].set_ylabel("Demographic Attribute", fontsize=12)
    axes[0].tick_params(axis='y', rotation=0, labelsize=12)
    axes[0].tick_params(axis='x', labelsize=11)

    # 2. Right Heatmap: Paraphrase Excess (Reality)
    sns.heatmap(df_pe, ax=axes[1], cmap=cmap, center=0, vmin=vmin, vmax=vmax, 
                annot=True, fmt=".3f", cbar=True, linewidths=.5, square=True,
                cbar_kws={'label': 'Absolute Shift Magnitude'},
                annot_kws={"size": 12, "weight": "bold"})
    axes[1].set_title("Corrected Reality (Paraphrase Excess Mean)", fontsize=14, pad=15)
    axes[1].tick_params(axis='x', labelsize=11)

    # Polish and layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure this points to your specific JSON file location
    json_path = "phase0_results/publication_metrics_n200_t0.0.json"
    generate_deconfounding_heatmap(json_path, task="esi", format_type="EXP")