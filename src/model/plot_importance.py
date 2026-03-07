import joblib
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# 🛑 THE FIX: We import the blueprint of our custom class 
# and explicitly link it to the __main__ namespace so joblib can find it.
from src.model.train_lgb import M5FeatureEngineer
sys.modules['__main__'].M5FeatureEngineer = M5FeatureEngineer

def plot_feature_importance(model_path):
    print(f"🔍 Loading model from {model_path}...")
    
    # 1. Load the artifact
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = artifact["metadata"]["features"]
    
    # 2. Extract importances
    importances = model.feature_importances_
    
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance (Split)": importances
    })
    df_imp = df_imp.sort_values(by="Importance (Split)", ascending=False)
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="Importance (Split)", 
        y="Feature", 
        data=df_imp, 
        palette="viridis",
        hue="Feature",
        legend=False
    )
    
    plt.title("LightGBM Feature Importance (M5 Dataset)", fontsize=14, pad=15)
    plt.xlabel("Number of Times Used to Split Trees", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    
    # 4. Save the plot
    output_file = "feature_importance.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ Feature importance chart saved to {output_file}")
    
    # Print top 5 to terminal
    print("\n🏆 Top 5 Features:")
    print(df_imp.head(5).to_string(index=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the .joblib file")
    args = parser.parse_args()
    
    plot_feature_importance(args.model_path)

if __name__ == "__main__":
    main()