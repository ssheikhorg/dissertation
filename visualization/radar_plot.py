import plotly.express as px

def plot_radar(dataframe, metrics, models):
    fig = px.line_polar(
        dataframe,
        r='value',
        theta='metric',
        color='model',
        line_close=True,
        template='plotly_dark'
    )
    fig.update_layout(title='Model Performance Comparison')
    return fig