import pandas as pd
from visualization.bar_plots import create_grouped_barplot, metric_comparison_bars, stacked_bias_bars

# Grouped bar plot
create_grouped_barplot(
    data=data,
    x_column='dataset',
    y_column='accuracy',
    hue_column='model',
    title='Accuracy Comparison Across Datasets'
)

# Metric comparison
metric_comparison_bars(
    results_df=data,
    metrics=['accuracy', 'toxicity'],
    title='Performance Metrics Comparison'
)

# Stacked bias plot (requires different data structure)
bias_breakdown = pd.DataFrame({
    'model': ['GPT-4', 'Claude', 'LLaMA'],
    'Gender': [0.15, 0.22, 0.18],
    'Racial': [0.08, 0.14, 0.11],
    'Political': [0.20, 0.17, 0.15]
})

stacked_bias_bars(
    results_df=bias_breakdown,
    bias_types=['Gender', 'Racial', 'Political']
)