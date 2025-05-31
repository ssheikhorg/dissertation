import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional
import numpy as np


def create_heatmap(
        data: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        value_column: str,
        title: str = "Model Performance Heatmap",
        cmap: str = "YlOrRd",
        figsize: tuple = (12, 8),
        annot: bool = True,
        normalize: bool = False,
        save_path: Optional[str] = None
) -> None:
    """
    Creates a heatmap from evaluation results.

    Args:
        data: DataFrame containing evaluation results
        x_axis: Column name for x-axis categories
        y_axis: Column name for y-axis categories
        value_column: Column name containing values to visualize
        title: Plot title
        cmap: Color map scheme
        figsize: Figure dimensions
        annot: Whether to display values in cells
        normalize: Whether to normalize values 0-1
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)

    # Pivot data for heatmap format
    heatmap_data = data.pivot_table(
        index=y_axis,
        columns=x_axis,
        values=value_column,
        aggfunc=np.mean
    )

    if normalize:
        heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    # Create heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        linewidths=.5,
        cbar_kws={'label': value_column}
    )

    plt.title(title, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def bias_heatmap(
        results_df: pd.DataFrame,
        model_names: List[str],
        bias_types: List[str],
        save_path: Optional[str] = None
) -> None:
    """
    Specialized heatmap for visualizing bias scores across models and bias categories.

    Args:
        results_df: DataFrame containing bias evaluation results
        model_names: List of model names to include
        bias_types: List of bias categories to evaluate
        save_path: Optional path to save the figure
    """
    bias_data = results_df[results_df['model'].isin(model_names)]

    plt.figure(figsize=(12, 8))

    # Prepare data - average scores per model and bias type
    heatmap_data = bias_data.groupby(['model', 'bias_type'])['score'].mean().unstack()

    # Reindex to ensure all bias types are shown
    heatmap_data = heatmap_data.reindex(columns=bias_types)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        linewidths=.5,
        cbar_kws={'label': 'Bias Score (higher = more biased)'}
    )

    plt.title('Bias Analysis Across Models', pad=20)
    plt.xlabel('Bias Type')
    plt.ylabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
