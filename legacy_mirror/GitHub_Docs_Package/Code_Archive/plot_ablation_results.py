import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "training_logs_ablation.csv")
TEST_RES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "final_test_results.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def plot_ablation_results():
    if not os.path.exists(LOG_PATH):
        print(f"Error: Log file not found at {LOG_PATH}")
        return
        
    df = pd.read_csv(LOG_PATH)
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 1. Loss Curves Comparison
    plt.figure(figsize=(12, 6))
    models = df['model'].unique()
    
    for model in models:
        subset = df[df['model'] == model]
        # Align epochs for Active Learning (it continues)
        # But here we just plot raw epochs per phase
        plt.plot(subset['epoch'], subset['val_loss'], label=f"{model} (Val)", linewidth=2)
        
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title("Training Dynamics: Ablation Study")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_loss_curves.png"), dpi=300)
    plt.close()
    
    # 2. Final Performance Bar Chart (MAE) - USING TEST SET METRICS
    if os.path.exists(TEST_RES_PATH):
        print(f"Loading Test Set metrics from {TEST_RES_PATH}")
        test_df = pd.read_csv(TEST_RES_PATH)
        
        # Prepare data for plotting
        final_metrics = []
        for index, row in test_df.iterrows():
            final_metrics.append({
                'Model': row['model'],
                'Metric': 'MAE Size',
                'Error (cm)': row['mae_size']
            })
            final_metrics.append({
                'Model': row['model'],
                'Metric': 'MAE Depth',
                'Error (cm)': row['mae_depth']
            })
            
        melted = pd.DataFrame(final_metrics)
        title_suffix = "(Independent Test Set)"
    else:
        print("Warning: Test Set metrics not found, falling back to Validation Best metrics.")
        # Fallback to Validation Best (Old logic)
        final_metrics = []
        for model in models:
            subset = df[df['model'] == model]
            best_epoch = subset.loc[subset['val_loss'].idxmin()]
            final_metrics.append({
                'Model': model,
                'Metric': 'MAE Size',
                'Error (cm)': best_epoch['val_mae_size']
            })
            final_metrics.append({
                'Model': model,
                'Metric': 'MAE Depth',
                'Error (cm)': best_epoch['val_mae_depth']
            })
        melted = pd.DataFrame(final_metrics)
        title_suffix = "(Validation Set)"
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="Model", y="Error (cm)", hue="Metric", palette="viridis")
    plt.title(f"Ablation Study: Prediction Error {title_suffix}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ablation_metrics_bar.png"), dpi=300)
    plt.close()
    
    print("Ablation charts generated.")

if __name__ == "__main__":
    plot_ablation_results()
