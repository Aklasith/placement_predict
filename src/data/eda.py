import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import validated ingestion function
from ingest import load_data


def perform_eda():

    # 1. Load Data
    DATA_PATH = os.path.join("src", "data", "raw_placement_data.csv")
    df = load_data(DATA_PATH)

    print("\n" + "=" * 50)
    print("--- 1. DATASET DIMENSIONS ---")
    print(f"Total Rows (Samples): {df.shape[0]}")
    print(f"Total Columns (Metrics): {df.shape[1]}")

    # 2. Feature Names & Data Types
    print("\n" + "=" * 50)
    print("--- 2. FEATURE NAMES & DATA TYPES ---")
    print(df.dtypes)

    # 3. Missing Values & Duplicates
    print("\n" + "=" * 50)
    print("--- 3. MISSING VALUES & DUPLICATES ---")

    missing_vals = df.isnull().sum()

    if missing_vals.sum() > 0:
        print("Missing Values per Column:")
        print(missing_vals[missing_vals > 0])
    else:
        print("No missing values found.")

    duplicates = df.duplicated().sum()
    print(f"Duplicate Records Count: {duplicates}")

    # 4. Summary Statistics
    print("\n" + "=" * 50)
    print("--- 4. SUMMARY STATISTICS ---")
    print(df.describe())

    # 5. Class Imbalance Analysis
    print("\n" + "=" * 50)
    print("--- 5. CLASS IMBALANCE ANALYSIS ---")

    if "placement_status" in df.columns:
        class_counts = df["placement_status"].value_counts()
        class_percentages = (
            df["placement_status"].value_counts(normalize=True) * 100
        )

        print("Placement Status Counts:")
        print(class_counts)

        print("\nPlacement Status Percentages:")
        print(class_percentages)
    else:
        print("Target column 'placement_status' not found.")

    # 6. Visualizations & Outlier Analysis
    print("\n" + "=" * 50)
    print("--- 6. GENERATING VISUALIZATIONS ---")

    sns.set_theme(style="whitegrid")

    os.makedirs("reports/figures", exist_ok=True)

    # A. Correlation Heatmap
    numerical_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(10, 8))
    corr_matrix = numerical_df.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Feature Correlation Matrix Heatmap")
    plt.tight_layout()

    plt.savefig("reports/figures/correlation_heatmap.png")
    plt.close()

    print("-> Saved correlation heatmap")

    # B. CGPA vs Salary Scatter Plot
    if "cgpa" in df.columns and "salary_package_lpa" in df.columns:

        plt.figure(figsize=(8, 6))

        sns.scatterplot(
            data=df,
            x="cgpa",
            y="salary_package_lpa",
            hue="placement_status",
            alpha=0.7
        )

        plt.title("CGPA vs Salary Package")
        plt.tight_layout()

        plt.savefig("reports/figures/scatter_cgpa_salary.png")
        plt.close()

        print("-> Saved scatter plot")

    # C. Pair Plot
    pairplot_cols = [
        "cgpa",
        "backlogs",
        "communication_skills",
        "internships",
        "salary_package_lpa"
    ]

    valid_pair_cols = [
        col for col in pairplot_cols
        if col in df.columns
    ]

    if len(valid_pair_cols) > 1:

        pp = sns.pairplot(
            df[valid_pair_cols],
            diag_kind="kde",
            corner=True
        )

        pp.fig.suptitle(
            "Pairwise Relationships of Key Numerical Features",
            y=1.02
        )

        pp.savefig("reports/figures/pairplot_features.png")
        plt.close()

        print("-> Saved pair plot")

    # D. Outlier Boxplots
    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=numerical_df,
        orient="h"
    )

    plt.title("Outlier Identification via Boxplots")
    plt.tight_layout()

    plt.savefig("reports/figures/outliers_boxplot.png")
    plt.close()

    print("-> Saved outlier boxplot")

    print("\nEDA Execution Complete!")
    print("Visualizations stored in reports/figures/")


if __name__ == "__main__":
    perform_eda()