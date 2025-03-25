import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Parse the log data
log_data = """
I0227 04:28:20.118471       1 gateway.go:564] ** latency metrics, hash(request), ab4ada31, selectedpod, 10.0.1.51:8000, ttft, 219, tpot, 17, e2e, 6702, numInputTokens, 421, numOutputTokens, 372, numTotalTokens, 793
I0227 04:28:29.419535       1 gateway.go:564] ** latency metrics, hash(request), ab4ada31, selectedpod, 10.0.1.51:8000, ttft, 78, tpot, 17, e2e, 6570, numInputTokens, 421, numOutputTokens, 372, numTotalTokens, 793
I0227 04:28:38.499986       1 gateway.go:564] ** latency metrics, hash(request), ab4ada31, selectedpod, 10.0.1.51:8000, ttft, 72, tpot, 17, e2e, 6555, numInputTokens, 421, numOutputTokens, 372, numTotalTokens, 793
I0227 04:28:57.675974       1 gateway.go:564] ** latency metrics, hash(request), 20fc7f6d, selectedpod, 10.0.1.26:8000, ttft, 259, tpot, 18, e2e, 5426, numInputTokens, 1396, numOutputTokens, 288, numTotalTokens, 1684
I0227 04:29:05.470685       1 gateway.go:564] ** latency metrics, hash(request), 20fc7f6d, selectedpod, 10.0.1.26:8000, ttft, 79, tpot, 18, e2e, 5249, numInputTokens, 1396, numOutputTokens, 288, numTotalTokens, 1684
I0227 04:29:13.253352       1 gateway.go:564] ** latency metrics, hash(request), 20fc7f6d, selectedpod, 10.0.1.26:8000, ttft, 78, tpot, 18, e2e, 5244, numInputTokens, 1396, numOutputTokens, 288, numTotalTokens, 1684
I0227 04:34:47.879316       1 gateway.go:564] ** latency metrics, hash(request), 24508ab5, selectedpod, 10.0.1.53:8000, ttft, 349, tpot, 18, e2e, 7048, numInputTokens, 2245, numOutputTokens, 363, numTotalTokens, 2608
I0227 04:34:57.611844       1 gateway.go:564] ** latency metrics, hash(request), 24508ab5, selectedpod, 10.0.1.53:8000, ttft, 88, tpot, 18, e2e, 6784, numInputTokens, 2245, numOutputTokens, 363, numTotalTokens, 2608
I0227 04:35:06.944500       1 gateway.go:564] ** latency metrics, hash(request), 24508ab5, selectedpod, 10.0.1.53:8000, ttft, 82, tpot, 18, e2e, 6779, numInputTokens, 2245, numOutputTokens, 363, numTotalTokens, 2608
"""

# Parse the log data into a structured format
rows = []
for line in log_data.strip().split('\n'):
    parts = line.split(', ')
    
    # Extract the required data
    hash_value = parts[2]
    ttft = int(parts[6])
    tpot = int(parts[8])
    e2e = int(parts[10])
    input_tokens = int(parts[12])
    output_tokens = int(parts[14])
    total_tokens = int(parts[16])
    
    rows.append({
        'hash': hash_value,
        'ttft': ttft,
        'tpot': tpot,
        'e2e': e2e,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens
    })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Group by hash to identify unique requests
request_groups = df.groupby('hash')

# Create named run columns (1st Run, 2nd Run, 3rd Run)
request_data = []
for hash_value, group in request_groups:
    group_sorted = group.reset_index(drop=True)
    
    # Sort the group by index to ensure correct order
    for i in range(len(group_sorted)):
        run_name = f"{i+1}{'st' if i == 0 else 'nd' if i == 1 else 'rd'} Run"
        row_data = group_sorted.iloc[i].to_dict()
        row_data['run'] = run_name
        row_data['request_name'] = f"{row_data['input_tokens']} tokens ({hash_value[:6]})"
        request_data.append(row_data)

# Create a new DataFrame with run information
result_df = pd.DataFrame(request_data)

# Configure plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create figure with 3 rows for TTFT, TPOT, and E2E
plt.figure(figsize=(8, 9))
gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.6)

# Colors for the bars
colors = ['#8884d8', '#82ca9d', '#ff8042']
metrics = ['ttft', 'tpot', 'e2e']
metric_names = ['Time to First Token (TTFT)', 'Time Per Output Token (TPOT)', 'End-to-End Latency (E2E)']

# Create subplots for each metric
for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    ax = plt.subplot(gs[i])
    
    # Set up plot
    ax.set_title(metric_name, fontsize=12, pad=10)
    ax.set_ylabel('ms', fontsize=12)
    
    # Get unique request names
    request_names = result_df['request_name'].unique()
    
    # Width of each bar group
    width = 0.25
    
    # Position of bars
    x = np.arange(len(request_names))
    
    # Create bars for each run
    for j, run_num in enumerate([0, 1, 2]):
        run_data = result_df[result_df['run'] == f"{j+1}{'st' if j == 0 else 'nd' if j == 1 else 'rd'} Run"]
        
        # Ensure data is in the same order as request_names
        values = []
        for req in request_names:
            val = run_data[run_data['request_name'] == req][metric].values
            values.append(val[0] if len(val) > 0 else 0)
        
        # Plot bars
        pos = x + (j - 1) * width
        bars = ax.bar(pos, values, width, label=f"{j+1}{'st' if j == 0 else 'nd' if j == 1 else 'rd'} Run", color=colors[j])
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{value}', ha='center', va='bottom', fontsize=9)
    
    # Set x-axis labels and ticks
    ax.set_xticks(x)
    ax.set_xticklabels(request_names, rotation=0)
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Start y-axis at 0
    ax.set_ylim(bottom=0)
    
    # Special handling for TPOT plot to ensure text labels fit
    if metric == 'tpot':
        # Set a fixed y-axis limit that's high enough for the labels
        ax.set_ylim(top=40)  # Set to 40 to leave room for the "18" labels
    else:
        # For other metrics, adjust y-axis limit dynamically to make room for labels
        current_ymax = ax.get_ylim()[1]
        ax.set_ylim(top=current_ymax * 1.1)
    
    # Add grid lines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(pad=1.5)
plt.savefig('latency_metrics.pdf', bbox_inches='tight')
plt.show()

# Create a summary table
print("\nSummary Table:")
summary_table = result_df.pivot_table(
    index=['request_name', 'input_tokens', 'output_tokens'],
    columns='run',
    values=['ttft', 'tpot', 'e2e'],
    aggfunc='first'
)
print(summary_table)