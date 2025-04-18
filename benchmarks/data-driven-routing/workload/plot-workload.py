import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_metrics(data):
    data.sort(key=lambda x: x['timestamp'])
    
    time_windows = defaultdict(lambda: {'requests': 0, 'prompt_tokens': 0})
    ms_to_s = 1000  # Convert milliseconds to seconds
    
    for entry in data:
        timestamp_ms = entry['timestamp']
        window_key = int(timestamp_ms // ms_to_s)  # Convert to seconds and use as key
        
        request_count = len(entry['requests'])
        time_windows[window_key]['requests'] += request_count
        
        prompt_tokens = sum(request['Prompt Length'] for request in entry['requests'])
        time_windows[window_key]['prompt_tokens'] += prompt_tokens
    
    times = sorted(time_windows.keys())
    rps = [time_windows[t]['requests'] for t in times]
    prompt_tokens_ps = [time_windows[t]['prompt_tokens'] for t in times]
    
    # Normalize times to be relative to the start time
    start_time = times[0] if times else 0
    times = [t - start_time for t in times]
    
    return times, rps, prompt_tokens_ps

def create_plot(times, rps, prompt_tokens_ps, output_file):
    # Set the font family and use a clean style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    })
    
    # Use a cleaner style
    plt.style.use('seaborn-v0_8-paper')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=300)
    
    # Plot RPS - removed markers
    color1 = '#0072B2'  # Professional blue
    ax1.plot(times, rps, color=color1, linewidth=2.5, label='Requests/Sec')
    ax1.set_ylabel('Requests per Second', fontweight='bold')
    ax1.tick_params(axis='both', which='major', width=1.5)
    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(frameon=True, loc='upper right', framealpha=0.9)
    
    # Show x-ticks on the first subplot as well
    ax1.tick_params(labelbottom=True)
    ax1.set_xlabel('Time (seconds since start)', fontweight='bold', labelpad=10)
    
    # Add light background shading to highlight data
    ax1.set_facecolor('#F8F8F8')
    
    # Plot Prompt Tokens per Second - removed markers
    color2 = '#D55E00'  # Professional orange/red
    ax2.plot(times, prompt_tokens_ps, color=color2, linewidth=2.5, label='Tokens/Sec')
    ax2.set_xlabel('Time (seconds since start)', fontweight='bold')
    ax2.set_ylabel('Prompt Tokens per Second', fontweight='bold')
    ax2.tick_params(axis='both', which='major', width=1.5)
    ax2.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(frameon=True, loc='upper right', framealpha=0.9)
    ax2.set_facecolor('#F8F8F8')
    
    # Add annotations to highlight key points
    avg_rps = np.mean(rps) if len(rps) > 0 else 0
    avg_tokens = np.mean(prompt_tokens_ps) if len(prompt_tokens_ps) > 0 else 0
    max_rps = max(rps) if len(rps) > 0 else 0
    max_tokens = max(prompt_tokens_ps) if len(prompt_tokens_ps) > 0 else 0
    ax1.axhline(avg_rps, color='gray', linestyle='--', linewidth=1.5, label=f'Avg RPS: {avg_rps:.2f}')
    ax2.axhline(avg_tokens, color='gray', linestyle='--', linewidth=1.5, label=f'Avg Tokens/Sec: {avg_tokens:.2f}')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax1.set_xlim(left=0)  # Ensure x-axis starts at 0
    ax2.set_xlim(left=0)  # Ensure x-axis starts at 0
    ax1.set_ylim(bottom=0)  # Ensure y-axis starts at 0 for RPS
    ax2.set_ylim(bottom=0)  # Ensure y-axis starts at 0 for Tokens/Sec
    fig.suptitle('Workload Performance Analysis', fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    # Save figure
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Plot saved to {output_file}")

def main(input_file):
    output_file = f'{input_file.split('.')[0]}.pdf'
    
    data = parse_jsonl(input_file)
    times, rps, prompt_tokens_ps = calculate_metrics(data)
    
    create_plot(times, rps, prompt_tokens_ps, output_file)

import sys
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        assert False, "Input file path is required as an argument."
    main(input_file)