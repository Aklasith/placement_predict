import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.ingest import load_data


def train_linear_regression_ls():

    # 1. Load Data
    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    df = load_data(DATA_PATH)

    # 2. Remove rows with missing required values
    feature_cols = [
        "cgpa",
        "communication_skill_score"
    ]

    target_col = "salary_package_lpa"

    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    # 3. Define X and y
    X_raw = df_clean[feature_cols].values
    y = df_clean[target_col].values.reshape(-1, 1)

    N = X_raw.shape[0]

    print(
        f"Loaded {N} data points "
        f"with input dimension L = {X_raw.shape[1]} "
        f"and output dimension M = {y.shape[1]}"
    )

    # 4. Add intercept/bias column
    # X_design = [1, cgpa, communication_skill_score]
    X_design = np.hstack(
        [
            np.ones((N, 1)),
            X_raw
        ]
    )

    # 5. Standard Least Squares / Normal Equation
    # w = (X^T X)^-1 X^T y

    XT_X = np.dot(
        X_design.T,
        X_design
    )

    try:
        XT_X_inv = np.linalg.inv(XT_X)

    except np.linalg.LinAlgError:
        print("Matrix is singular. Using pseudo-inverse.")
        XT_X_inv = np.linalg.pinv(XT_X)

    XT_y = np.dot(
        X_design.T,
        y
    )

    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )

    # 6. Display model parameters
    print("\n" + "=" * 50)
    print("--- OPTIMAL MODEL PARAMETERS ---")

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.4f}"
    )

    print(
        f"Coefficient for CGPA (w1): "
        f"{w_optimal[1, 0]:.4f}"
    )

    print(
        f"Coefficient for Communication Skills (w2): "
        f"{w_optimal[2, 0]:.4f}"
    )

    # 7. Calculate predictions
    y_pred = np.dot(
        X_design,
        w_optimal
    )

    # 8. Calculate error
    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )

    print(
        f"Minimized Error (E_w): "
        f"{E_w:.4f}"
    )

    # 9. Create 3D regression plane
    os.makedirs(
        "reports/figures",
        exist_ok=True
    )

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        projection="3d"
    )

    # Actual data points
    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y.flatten(),
        alpha=0.6,
        label="Actual Data Points"
    )

    # Create meshgrid
    x1_surf = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        20
    )

    x2_surf = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        20
    )

    x1_mesh, x2_mesh = np.meshgrid(
        x1_surf,
        x2_surf
    )

    # Regression plane
    y_mesh = (
        w_optimal[0, 0]
        + w_optimal[1, 0] * x1_mesh
        + w_optimal[2, 0] * x2_mesh
    )

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        alpha=0.3,
        edgecolor="none"
    )

    ax.set_xlabel("CGPA")
    ax.set_ylabel("Communication Skill Score")
    ax.set_zlabel("Salary Package LPA")

    ax.set_title(
        "Linear Regression via Standard Least Squares"
    )

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "linear_regression_3d_plane.png"
    )

    plt.savefig(output_path)
    plt.close()

    print(
        f"\n-> Successfully saved regression plot to "
        f"{output_path}"
    )


if __name__ == "__main__":
    train_linear_regression_ls()