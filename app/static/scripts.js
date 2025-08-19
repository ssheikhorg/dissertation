// Prepare data for bar chart visualization
function prepare_bar_chart_data(general_df, metric, models = null) {
    if (!general_df || general_df.length === 0) {
        return { error: "No data available for visualization" };
    }

    // Filter by specific models if provided
    let filteredData = general_df;
    if (models && models.length > 0) {
        filteredData = general_df.filter(item => models.includes(item.model));
    }

    // Group by model and calculate average for the selected metric
    const modelAverages = {};
    filteredData.forEach(item => {
        if (!modelAverages[item.model]) {
            modelAverages[item.model] = { sum: 0, count: 0 };
        }
        modelAverages[item.model].sum += item[metric] || 0;
        modelAverages[item.model].count += 1;
    });

    // Calculate averages
    const chartData = {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: metric.replace(/_/g, ' ').toUpperCase(),
                data: [],
                backgroundColor: [],
                borderColor: [],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `${metric.replace(/_/g, ' ').toUpperCase()} Comparison`,
                    font: { size: 16 }
                },
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: metric.replace(/_/g, ' ').toUpperCase()
                    }
                }
            }
        }
    };

    // Assign colors based on metric value (red for high hallucination, green for good metrics)
    Object.keys(modelAverages).forEach(model => {
        const average = modelAverages[model].sum / modelAverages[model].count;
        chartData.data.labels.push(model);
        chartData.data.datasets[0].data.push(average);

        // Color coding based on metric type and value
        if (metric.includes('hallucination') || metric.includes('toxicity')) {
            // Red scale for negative metrics (higher is worse)
            const intensity = Math.min(1, average * 3); // Scale for better color distribution
            chartData.data.datasets[0].backgroundColor.push(`rgba(231, 76, 60, ${0.6 + intensity * 0.4})`);
            chartData.data.datasets[0].borderColor.push(`rgba(231, 76, 60, 1)`);
        } else {
            // Green scale for positive metrics (higher is better)
            const intensity = Math.min(1, average * 1.5);
            chartData.data.datasets[0].backgroundColor.push(`rgba(46, 204, 113, ${0.6 + intensity * 0.4})`);
            chartData.data.datasets[0].borderColor.push(`rgba(46, 204, 113, 1)`);
        }
    });

    return chartData;
}

// Prepare data for radar chart visualization
function prepare_radar_data(general_df, models, metrics) {
    if (!general_df || general_df.length === 0) {
        return { error: "No data available for visualization" };
    }

    // Default metrics if not specified
    if (!metrics || metrics.length === 0) {
        metrics = ['hallucination_rate', 'accuracy', 'fact_score', 'toxicity_score'];
    }

    // Filter by specific models if provided, otherwise use all available
    const selectedModels = models && models.length > 0 ? models : [...new Set(general_df.map(item => item.model))];

    const chartData = {
        type: 'radar',
        data: {
            labels: metrics.map(metric => metric.replace(/_/g, ' ').toUpperCase()),
            datasets: []
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Radar Chart',
                    font: { size: 16 }
                },
                legend: {
                    position: 'top',
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    }
                }
            }
        }
    };

    // Colors for different models
    const colors = [
        'rgba(255, 99, 132, 0.6)', // Red
        'rgba(54, 162, 235, 0.6)',  // Blue
        'rgba(255, 206, 86, 0.6)',  // Yellow
        'rgba(75, 192, 192, 0.6)',  // Green
        'rgba(153, 102, 255, 0.6)', // Purple
        'rgba(255, 159, 64, 0.6)'   // Orange
    ];

    // Calculate averages for each model and metric
    selectedModels.forEach((model, index) => {
        const modelData = general_df.filter(item => item.model === model);
        const averages = [];

        metrics.forEach(metric => {
            const values = modelData.map(item => item[metric] || 0).filter(val => val !== null);
            const average = values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : 0;

            // For negative metrics like hallucination_rate, invert for radar (so center is good)
            if (metric.includes('hallucination') || metric.includes('toxicity')) {
                averages.push(1 - average); // Invert so lower is better (closer to center)
            } else {
                averages.push(average);
            }
        });

        chartData.data.datasets.push({
            label: model,
            data: averages,
            backgroundColor: colors[index % colors.length],
            borderColor: colors[index % colors.length].replace('0.6', '1'),
            borderWidth: 2,
            pointBackgroundColor: colors[index % colors.length].replace('0.6', '1'),
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: colors[index % colors.length].replace('0.6', '1')
        });
    });

    return chartData;
}

// Prepare data for scatter plot visualization
function prepare_scatter_data(general_df, primaryMetric, secondaryMetric = null) {
    if (!general_df || general_df.length === 0) {
        return { error: "No data available for visualization" };
    }

    // Default secondary metric if not specified
    if (!secondaryMetric) {
        secondaryMetric = primaryMetric.includes('hallucination') ? 'accuracy' : 'hallucination_rate';
    }

    const chartData = {
        type: 'scatter',
        data: {
            datasets: []
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `${primaryMetric.replace(/_/g, ' ').toUpperCase()} vs ${secondaryMetric.replace(/_/g, ' ').toUpperCase()}`,
                    font: { size: 16 }
                },
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: primaryMetric.replace(/_/g, ' ').toUpperCase()
                    },
                    beginAtZero: true,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: secondaryMetric.replace(/_/g, ' ').toUpperCase()
                    },
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    };

    // Group data by model
    const models = [...new Set(general_df.map(item => item.model))];
    const colors = [
        'rgba(255, 99, 132, 0.8)', // Red
        'rgba(54, 162, 235, 0.8)',  // Blue
        'rgba(255, 206, 86, 0.8)',  // Yellow
        'rgba(75, 192, 192, 0.8)',  // Green
        'rgba(153, 102, 255, 0.8)', // Purple
        'rgba(255, 159, 64, 0.8)'   // Orange
    ];

    models.forEach((model, index) => {
        const modelData = general_df.filter(item => item.model === model);
        const points = [];

        modelData.forEach(item => {
            if (item[primaryMetric] !== undefined && item[secondaryMetric] !== undefined) {
                points.push({
                    x: item[primaryMetric],
                    y: item[secondaryMetric]
                });
            }
        });

        if (points.length > 0) {
            chartData.data.datasets.push({
                label: model,
                data: points,
                backgroundColor: colors[index % colors.length],
                borderColor: colors[index % colors.length].replace('0.8', '1'),
                borderWidth: 1,
                pointRadius: 8,
                pointHoverRadius: 10
            });
        }
    });

    return chartData;
}

// Enhanced renderVisualization function that uses the above functions
function renderVisualization(data) {
    if (data.error) {
        document.getElementById('visualization-container').innerHTML = `
            <div class="error">Error: ${data.error}</div>
        `;
        return;
    }

    const vizType = document.getElementById('visualization-type').value;
    const metric = document.getElementById('metric-select').value;

    let chartData;

    switch(vizType) {
        case 'bar':
            chartData = prepare_bar_chart_data(data.general_df, metric, data.models);
            break;
        case 'radar':
            chartData = prepare_radar_data(data.general_df, data.models, [metric]);
            break;
        case 'scatter':
            chartData = prepare_scatter_data(data.general_df, metric);
            break;
        default:
            document.getElementById('visualization-container').innerHTML = `
                <div class="error">Error: Unknown visualization type</div>
            `;
            return;
    }

    if (chartData.error) {
        document.getElementById('visualization-container').innerHTML = `
            <div class="error">${chartData.error}</div>
        `;
        return;
    }

    // Render the chart using Chart.js (you'll need to include Chart.js in your project)
    renderChartWithChartJS(chartData);
}

// Function to render chart using Chart.js (you need to include Chart.js library)
function renderChartWithChartJS(chartData) {
    const ctx = document.getElementById('chart-canvas');

    // Create canvas if it doesn't exist
    if (!ctx) {
        const canvas = document.createElement('canvas');
        canvas.id = 'chart-canvas';
        canvas.width = 800;
        canvas.height = 500;
        document.getElementById('visualization-container').innerHTML = '';
        document.getElementById('visualization-container').appendChild(canvas);
    }

    // Render the chart
    try {
        new Chart(
            document.getElementById('chart-canvas'),
            chartData
        );
    } catch (error) {
        console.error('Chart rendering error:', error);
        document.getElementById('visualization-container').innerHTML = `
            <div class="error">Error rendering chart: ${error.message}</div>
            <div class="fallback-data">
                <h4>Data for ${chartData.options.plugins.title.text}</h4>
                <pre>${JSON.stringify(chartData.data, null, 2)}</pre>
            </div>
        `;
    }
}

// Update the setupVisualizationForm function to handle the new data structure
function setupVisualizationForm() {
    const vizForm = document.querySelector('#visualize-page form');
    if (vizForm) {
        vizForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const vizButton = this.querySelector('button[type="submit"]');
            const vizType = document.getElementById('visualization-type').value;
            const metric = document.getElementById('metric-select').value;

            // Show loading state
            vizButton.disabled = true;
            vizButton.textContent = 'Generating...';

            try {
                const response = await fetch('/api/visualize-data', {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });

                const data = await response.json();

                // Add the visualization type and metric to the data
                data.vizType = vizType;
                data.metric = metric;

                renderVisualization(data);
            } catch (error) {
                console.error('Visualization error:', error);
                document.getElementById('visualization-container').innerHTML = `
                    <div class="error">Error: Failed to generate visualization. Please try again.</div>
                `;
            } finally {
                // Restore button state
                vizButton.disabled = false;
                vizButton.textContent = 'Generate Visualization';
            }
        });
    }
}