import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Optional
import numpy as np


def create_grouped_barplot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    hue_column: str,
    title: str = "Model Performance Comparison",
    x_label: str = "",
    y_label: str = "Score",
    palette: str = "muted",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Creates a grouped bar plot for comparing models across different metrics or datasets.

    Args:
        data: DataFrame containing evaluation results
        x_column: Column for x-axis grouping
        y_column: Column for bar heights
        hue_column: Column for color grouping
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        palette: Color palette
        figsize: Figure dimensions
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)

    ax = sns.barplot(
        data=data,
        x=x_column,
        y=y_column,
        hue=hue_column,
        palette=palette,
        errorbar=None,
        estimator=np.mean,
    )

    plt.title(title, pad=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")

    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.legend(title=hue_column, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def metric_comparison_bars(
    results_df: pd.DataFrame,
    metrics: List[str],
    model_col: str = "model",
    title: str = "Model Performance Metrics Comparison",
    save_path: Optional[str] = None,
) -> None:
    """
    Creates a bar plot comparing multiple metrics across models.

    Args:
        results_df: DataFrame containing evaluation results
        metrics: List of metric columns to compare
        model_col: Name of column containing model names
        title: Plot title
        save_path: Optional path to save the figure
    """
    # Melt data for seaborn
    melted_data = results_df.melt(
        id_vars=[model_col], value_vars=metrics, var_name="metric", value_name="score"
    )

    plt.figure(figsize=(14, 7))

    ax = sns.barplot(
        data=melted_data,
        x="metric",
        y="score",
        hue=model_col,
        palette="Set2",
        errorbar=None,
    )

    plt.title(title, pad=20)
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=45)

    # Add value labels
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def stacked_bias_bars(
    results_df: pd.DataFrame,
    bias_types: List[str],
    model_col: str = "model",
    title: str = "Bias Composition Across Models",
    save_path: Optional[str] = None,
) -> None:
    """
    Creates a stacked bar plot showing bias composition across models.

    Args:
        results_df: DataFrame containing bias evaluation results
        bias_types: List of bias type columns
        model_col: Name of column containing model names
        title: Plot title
        save_path: Optional path to save the figure
    """
    # Calculate proportions
    bias_totals = results_df.groupby(model_col)[bias_types].sum()
    bias_proportions = bias_totals.div(bias_totals.sum(axis=1), axis=0)

    # Plot stacked bars
    ax = bias_proportions.plot(
        kind="bar", stacked=True, figsize=(12, 6), colormap="RdYlGn_r", title=title
    )

    plt.title(title, pad=20)
    plt.xlabel("Model")
    plt.ylabel("Proportion of Bias")
    plt.xticks(rotation=45)

    # Add legend outside plot
    plt.legend(title="Bias Types", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


from visualization.bar_plots import (
    create_grouped_barplot,
    metric_comparison_bars,
    stacked_bias_bars,
)

# Grouped bar plot
create_grouped_barplot(
    data=data,
    x_column="dataset",
    y_column="accuracy",
    hue_column="model",
    title="Accuracy Comparison Across Datasets",
)

# Metric comparison
metric_comparison_bars(
    results_df=data,
    metrics=["accuracy", "toxicity"],
    title="Performance Metrics Comparison",
)

# Stacked bias plot (requires different data structure)
bias_breakdown = pd.DataFrame(
    {
        "model": ["GPT-4", "Claude", "LLaMA"],
        "Gender": [0.15, 0.22, 0.18],
        "Racial": [0.08, 0.14, 0.11],
        "Political": [0.20, 0.17, 0.15],
    }
)

stacked_bias_bars(
    results_df=bias_breakdown, bias_types=["Gender", "Racial", "Political"]
)
