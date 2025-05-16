import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import json
from datetime import datetime
import logging
import training.preprocess as preprocess

def parse_strategy_name(filepath):
    """Extract the routing strategy name from the filepath."""
    # Pattern matches directories like "latency-prediction-based:none" from the path
    pattern = r"/([^/]+:[^/]+)/"
    match = re.search(pattern, filepath)
    if match:
        return match.group(1)
    else:
        # Fallback to the immediate parent directory if the pattern doesn't match
        return os.path.basename(os.path.dirname(filepath))

def find_log_files(base_dir):
    """Recursively find all filtered log CSV files in the base directory."""
    pattern = os.path.join(base_dir, "**", "filtered-aibrix-gateway-plugins.log.csv")
    return glob.glob(pattern, recursive=True)

def analyze_llm_inference_logs(df):
    """Process the dataframe to calculate basic statistics - modified from your existing function."""
    if df.empty:
        print("No valid data found in the log file.")
        return df
    
    # Calculate experiment duration
    if 'request_start_time' in df.columns and 'request_end_time' in df.columns:
        start_time = df['request_start_time'].min()
        end_time = df['request_end_time'].max()
        experiment_duration = (end_time - start_time) / 1000000
        print(f"Experiment duration: {experiment_duration:.2f} seconds")
    
    if 'selectedpod' in df.columns:
        df['selectedpod'] = df['selectedpod'].str.split(':').str[0]

    # Process other metrics as in the original function
    # The rest of the processing is the same as in your original function
    # This is a simplified version focusing on the metrics we need
    
    return df

def calculate_performance_metrics(df):
    """Calculate the performance metrics for a dataframe."""
    metrics = {}
    
    # Calculate TTFT metrics if available
    if 'ttft' in df.columns:
        metrics['avg_ttft'] = df['ttft'].mean()
        metrics['p99_ttft'] = df['ttft'].quantile(0.99)
    
    # Calculate TPOT metrics if available
    if 'avg_tpot' in df.columns:
        metrics['avg_tpot'] = df['avg_tpot'].mean()
        metrics['p99_tpot'] = df['avg_tpot'].quantile(0.99)
    
    # Calculate throughput
    if 'normalized_start_time' in df.columns and 'request_end_time' in df.columns:
        # Calculate duration in seconds
        total_duration = (df['request_end_time'].max() - df['request_start_time'].min()) / 1000000
        if total_duration > 0:
            metrics['throughput_rps'] = len(df) / total_duration
        else:
            metrics['throughput_rps'] = 0
    
    # Calculate output token throughput if available
    if 'normalized_start_time' in df.columns and 'numOutputTokens' in df.columns:
        total_output_tokens = df['numOutputTokens'].sum()
        if 'request_end_time' in df.columns and 'request_start_time' in df.columns:
            total_duration = (df['request_end_time'].max() - df['request_start_time'].min()) / 1000000
            if total_duration > 0:
                metrics['throughput_tps'] = total_output_tokens / total_duration
            else:
                metrics['throughput_tps'] = 0
    
    return metrics

def process_log_file(file_path):
    """Process a single log file and return its performance metrics."""
    try:
        print(f"Processing {file_path}...")
        df, json_columns = preprocess.parse_log_file(file_path)
        df = preprocess.parse_json_columns(df, json_columns)
        df = preprocess.normalize_time(df)
        df = analyze_llm_inference_logs(df)
        
        # Extract strategy name from the file path
        strategy = parse_strategy_name(file_path)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(df)
        metrics['strategy'] = strategy
        metrics['file_path'] = file_path
        metrics['num_requests'] = len(df)
        
        return metrics
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {'strategy': parse_strategy_name(file_path), 'error': str(e)}

def plot_routing_comparison(metrics_list):
    """Create bar charts comparing performance metrics across routing strategies."""
    if not metrics_list:
        print("No metrics to plot.")
        return
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort strategies by avg_ttft for consistent ordering
    if 'avg_ttft' in metrics_df.columns:
        strategy_order = metrics_df.sort_values('avg_ttft')['strategy'].tolist()
    else:
        strategy_order = metrics_df['strategy'].tolist()
    
    # Set up colors for each strategy
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(strategy_order)))
    color_dict = dict(zip(strategy_order, colors))
    
    # Create figure with the default size as requested
    fig, axes = plt.subplots(3, 2)
    fig.suptitle('Routing Strategy Performance Comparison', fontsize=12)
    
    # Plot 1: Average TTFT
    if 'avg_ttft' in metrics_df.columns:
        ax = axes[0, 0]
        plot_metric_bar(ax, metrics_df, 'avg_ttft', 'Average TTFT (ms)', 
                        strategy_order, color_dict)
    
    # Plot 2: P99 TTFT
    if 'p99_ttft' in metrics_df.columns:
        ax = axes[0, 1]
        plot_metric_bar(ax, metrics_df, 'p99_ttft', 'P99 TTFT (ms)', 
                        strategy_order, color_dict)
    
    # Plot 3: Average TPOT
    if 'avg_tpot' in metrics_df.columns:
        ax = axes[1, 0]
        plot_metric_bar(ax, metrics_df, 'avg_tpot', 'Average TPOT (ms)', 
                        strategy_order, color_dict)
    
    # Plot 4: P99 TPOT
    if 'p99_tpot' in metrics_df.columns:
        ax = axes[1, 1]
        plot_metric_bar(ax, metrics_df, 'p99_tpot', 'P99 TPOT (ms)', 
                        strategy_order, color_dict)
    
    # Plot 5: Throughput (Requests per Second)
    if 'throughput_rps' in metrics_df.columns:
        ax = axes[2, 0]
        plot_metric_bar(ax, metrics_df, 'throughput_rps', 'Throughput (Requests/sec)', 
                        strategy_order, color_dict)
    
    # Plot 6: Token Throughput
    if 'throughput_tps' in metrics_df.columns:
        ax = axes[2, 1]
        plot_metric_bar(ax, metrics_df, 'throughput_tps', 'Throughput (Tokens/sec)', 
                        strategy_order, color_dict)
    
    # Create a single shared legend for all plots
    # Only create a shared legend if not already created in the subplots
    handles = [plt.Rectangle((0,0), 1, 1, color=color_dict[s]) for s in strategy_order]
    legend_labels = [s if len(s) < 20 else s[:17]+'...' for s in strategy_order]
    
    # Place the legend below the entire figure in a horizontal layout
    fig.legend(handles, legend_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, 0.02),
              fontsize=8, ncol=min(3, len(strategy_order)), 
              title="Routing Strategies")
    
    # Adjust layout parameters to accommodate the legend
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.92, bottom=0.20, hspace=0.3, wspace=0.2)
    
    # Save the figure
    output_file = "routing_strategy_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_file}")
    
    plt.show()

def plot_metric_bar(ax, metrics_df, metric, title, strategy_order, color_dict):
    """Helper function to create a bar chart for a specific metric."""
    if metric not in metrics_df.columns:
        ax.text(0.5, 0.5, f"No data for {metric}", 
                horizontalalignment='center', verticalalignment='center')
        ax.set_title(title, fontsize=10)
        return
    
    # Sort by strategy order
    plot_data = metrics_df.set_index('strategy').loc[strategy_order, [metric]]
    
    # Use simple index numbers for x-axis
    bar_positions = np.arange(len(plot_data))
    
    # Create bar chart with bars
    bars = ax.bar(bar_positions, plot_data[metric], 
                  color=[color_dict[s] for s in strategy_order])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),  # 2 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7)  # Smaller font for annotations
    
    # Set chart titles and labels
    ax.set_title(title, fontsize=10, pad=2)
    ax.set_ylabel(title.split('(')[0].strip(), fontsize=8)
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel('')
    
    # Optimize y-axis ticks
    ax.tick_params(axis='y', labelsize=7)
    if len(ax.get_yticks()) > 5:
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    ax.grid(axis='y', alpha=0.3)

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_routing_strategies.py <base_directory>")
        print("Example: python compare_routing_strategies.py filtered_logs/chatbot-simulation")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    print(f"Searching for log files in {base_dir}...")
    
    log_files = find_log_files(base_dir)
    print(f"Found {len(log_files)} log files.")
    
    if not log_files:
        print(f"No log files found in {base_dir}")
        sys.exit(1)
    
    # Process each log file
    all_metrics = []
    for log_file in log_files:
        metrics = process_log_file(log_file)
        if metrics:
            all_metrics.append(metrics)
    
    # Plot the comparison
    plot_routing_comparison(all_metrics)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for metrics in all_metrics:
        strategy = metrics.get('strategy', 'Unknown')
        print(f"\nStrategy: {strategy}")
        for key, value in metrics.items():
            if key not in ['strategy', 'file_path']:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()