import plotly.express as px
import pandas as pd
from typing import Dict, List


def create_radar_plot(results: Dict[str, Dict], metrics: List[str]) -> None:
    plot_data = []
    for model, scores in results.items():
        for metric in metrics:
            plot_data.append(
                {"Model": model, "Metric": metric, "Score": scores[metric]}
            )

    df = pd.DataFrame(plot_data)
    fig = px.line_polar(
        df,
        r="Score",
        theta="Metric",
        color="Model",
        line_close=True,
        template="plotly_dark",
        title="Model Performance Comparison",
    )
    fig.show()
