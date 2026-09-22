import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_table_figures(json_path: str):
    path = Path(json_path)
    if not path.exists():
        print(f"File not found: {json_path}")
        return

    with open(path, "r") as f:
        data = json.load(f)

    demo_data = data.get("demographic_bias", {})
    models = list(demo_data.keys())
    attributes = ["ses", "race", "sex"]
    formats = ["EXP", "IMP", "PRX"]
    
    # Define the two tables
    tasks = [
        ("usmle", "Table 1a: USMLE Diagnostics (Experiment 1)"),
        ("esi", "Table 1b: ESI Triage (Experiment 1)")
    ]

    # Set up output directory
    output_dir = path.parent
    
    columns = ["Model", "Attr", "Format", "CFR", "AD", "Cohen's h", "McNemar p", "Excess Mean", "95% CI", "Verdict"]

    for task_key, table_title in tasks:
        rows = []
        
        for model in models:
            for attr in attributes:
                for fmt in formats:
                    # 1. Extract Naive Fairness Metrics
                    task_data = demo_data.get(model, {}).get(attr, {}).get(fmt, {}).get(task_key, {})
                    if not task_data:
                        continue

                    cfr = task_data.get("cfr", {}).get("cfr")
                    ad = task_data.get("ad", {}).get("ad")
                    mc = task_data.get("mcnemar", {})
                    p_val = mc.get("p_value")
                    coh_h = mc.get("cohens_h")

                    # 2. Extract Paraphrase Correction Metrics
                    pe_data = demo_data.get(model, {}).get(attr, {}).get("paraphrase_excess", {}).get(fmt, {}).get(task_key, {})
                    pe_mean = pe_data.get("mean")
                    ci_low = pe_data.get("ci_low")
                    ci_high = pe_data.get("ci_high")

                    # 3. Formatting Rules
                    cfr_str = f"{cfr:.3f}" if cfr is not None else "-"
                    ad_str = f"{ad:.3f}" if ad is not None else "-"
                    coh_str = f"{coh_h:.3f}" if coh_h is not None else "-"

                    # Rule A: Flag false positives (p < 0.05)
                    p_str = "-"
                    if p_val is not None:
                        if p_val < 0.05:
                            p_str = f"{p_val:.3f}*" # Asterisk flags significance in the PNG
                        else:
                            p_str = f"{p_val:.3f}"

                    # Rule B: The Zero-Crossing Rule
                    mean_str = f"{pe_mean:.3f}" if pe_mean is not None else "-"
                    ci_str = "-"
                    crosses_zero = False
                    
                    if ci_low is not None and ci_high is not None:
                        crosses_zero = (ci_low <= 0 <= ci_high)
                        base_ci = f"[{ci_low:.3f}, {ci_high:.3f}]"
                        ci_str = f"{base_ci} (Null)" if crosses_zero else base_ci

                    # Rule C: The Verdict Logic
                    verdict = "Null"
                    if p_val is not None and p_val < 0.05:
                        if crosses_zero:
                            verdict = "Confounded" # Benchmarks see bias, math proves it's rewriting noise
                        else:
                            verdict = "Significant" # Actual demographic bias

                    rows.append([
                        model, attr.upper(), fmt, cfr_str, ad_str, 
                        coh_str, p_str, mean_str, ci_str, verdict
                    ])
        
        if not rows:
            continue
            
        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        # --- EXPORT 1: Save as CSV for Excel/LaTeX import ---
        csv_path = output_dir / f"{task_key}_experiment_1_table.csv"
        df.to_csv(csv_path, index=False)
        
        # --- EXPORT 2: Save as Markdown file ---
        md_path = output_dir / f"{task_key}_experiment_1_table.md"
        with open(md_path, "w") as f:
            f.write(f"### {table_title}\n\n")
            f.write(df.to_markdown(index=False))
            
        # --- EXPORT 3: Render and save as a high-res PNG Figure ---
        # --- EXPORT 3: Render and save as a high-res PNG Figure ---
        # Increased figure width from 12 to 16 to give the columns room to breathe
        fig, ax = plt.subplots(figsize=(16, len(df) * 0.3 + 1.5))
        ax.axis('tight')
        ax.axis('off')
        
        # Draw the table
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8) # Slightly taller rows for better vertical spacing
        
        # THE FIX: Force matplotlib to auto-size the width of every column based on text length
        table.auto_set_column_width(col=list(range(len(df.columns))))
        
        # Bold the headers and shade PRX rows
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor("#e0e0e0") # Added a subtle header background
            elif row > 0 and df.iloc[row-1]["Format"] == "PRX":
                cell.set_facecolor("#f5f5f5") # Lighter grey for PRX rows
                
        plt.title(table_title, weight='bold', size=16, pad=20)
        
        png_path = output_dir / f"{task_key}_experiment_1_figure.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Point this to your generated metrics file
    generate_table_figures("phase0_results/publication_metrics_n200_t0.0.json")