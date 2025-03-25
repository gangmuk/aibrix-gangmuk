import json
import argparse
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def scale_to_target_avg(requests_per_second, target_avg_rps, original_avg_rps):
    """Scale requests to match target average RPS"""
    print(f"Original average RPS: {original_avg_rps}")
    print(f"Scale to target average RPS: {target_avg_rps}")
    scaled_requests = defaultdict(list)
    scaling_factor = target_avg_rps / original_avg_rps
    print(f"Scaling factor for average RPS: {scaling_factor:.3f}")
    
    for second, entries in requests_per_second.items():
        total_requests = len(entries)
        target_requests = int(total_requests * scaling_factor)
        
        if target_requests < total_requests:
            # Move excess requests to next second
            scaled_requests[second] = entries[:target_requests]
            scaled_requests[second + 1].extend(entries[target_requests:])
        else:
            scaled_requests[second] = entries
            
    return scaled_requests

def enforce_min_max_rps(requests_per_second, min_rps, max_rps):
    """Enforce min/max RPS bounds on each second"""
    print(f"Enforce min/max RPS bounds: {min_rps} - {max_rps}")
    bounded_requests = defaultdict(list)
    for second in sorted(requests_per_second.keys()):
        entries = requests_per_second[second]
        total_requests = len(entries)
        if min_rps is not None and total_requests < min_rps:
            # print(f"Second {second}: Too few requests ({total_requests}), minimum is {min_rps}")
            next_second = second + 1
            if next_second in requests_per_second:
                needed = int(min_rps) - total_requests
                next_entries = requests_per_second[next_second]
                if next_entries:
                    move_entries = next_entries[:needed]
                    requests_per_second[next_second] = next_entries[needed:]
                    entries.extend(move_entries)
        if max_rps is not None and total_requests > max_rps:
            print(f"Second {second}: Too many requests ({total_requests}), maximum is {max_rps}")
            bounded_requests[second] = entries[:int(max_rps)]
            bounded_requests[second + 1].extend(entries[int(max_rps):])
        else:
            bounded_requests[second] = entries
            
    return bounded_requests

def plot_rps_comparison(original_rps, scaled_rps, output_file=None):
    """Create a publication-quality plot comparing original and scaled RPS"""
    plt.figure(figsize=(12, 6))
    
    # Get time range
    all_seconds = sorted(set(list(original_rps.keys()) + list(scaled_rps.keys())))
    min_second = min(all_seconds)
    # Convert to relative time (starting from 0)
    time_points = [(s - min_second) for s in all_seconds]
    
    # Prepare data points
    original_values = [original_rps.get(s, 0) for s in all_seconds]
    scaled_values = [scaled_rps.get(s, 0) for s in all_seconds]
    
    # Create the plot
    plt.plot(time_points, original_values, 'b-', label='Original', linewidth=2, alpha=0.7)
    plt.plot(time_points, scaled_values, 'r-', label='Scaled', linewidth=2, alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Requests per Second', fontsize=12)
    plt.title('Workload RPS Comparison', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Add statistics as text
    orig_stats = f'Original - Avg: {np.mean(original_values):.1f}, Min: {min(original_values):.1f}, Max: {max(original_values):.1f}'
    scaled_stats = f'Scaled - Avg: {np.mean(scaled_values):.1f}, Min: {min(scaled_values):.1f}, Max: {max(scaled_values):.1f}'
    plt.text(0.02, 0.98, orig_stats + '\n' + scaled_stats,
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Ensure integer y-axis ticks
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output file specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    fn = 'rps_comparison.pdf'
    plt.savefig(fn, bbox_inches='tight')
    print(f"Plot saved to {fn}")
    plt.show()

def process_jsonl_rps(input_file, output_file, target_avg_rps=None, min_rps=None, max_rps=None):
    # First pass: gather timestamps and calculate RPS in 1-second buckets
    requests_per_second = defaultdict(list)
    
    print("Reading input file...")
    with open(input_file, 'r') as fin:
        for line in fin:
            try:
                data = json.loads(line.strip())
                second = data['timestamp'] // 1000  # Convert to seconds
                requests_per_second[second].append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON: {e}")
                continue
    
    if not requests_per_second:
        raise ValueError("No valid data points found")
    
    # Calculate original RPS statistics
    original_rps_values = []
    start_second = min(requests_per_second.keys())
    end_second = max(requests_per_second.keys())
    
    # Store original RPS data
    original_rps_data = {}
    for second in range(start_second, end_second + 1):
        rps = len(requests_per_second[second])
        original_rps_values.append(rps)
        original_rps_data[second] = rps
    
    original_avg_rps = statistics.mean(original_rps_values)
    print(f"\nOriginal RPS statistics:")
    print(f"  Average: {original_avg_rps:.2f}")
    print(f"  Min: {min(original_rps_values):.2f}")
    print(f"  Max: {max(original_rps_values):.2f}")
    
    # First transformation: Scale to target average RPS
    if target_avg_rps is not None:
        print("\nScaling to target average RPS...")
        requests_per_second = scale_to_target_avg(requests_per_second, target_avg_rps, original_avg_rps)
    
    # Second transformation: Enforce min/max RPS bounds
    if min_rps is not None or max_rps is not None:
        print("\nEnforcing min/max RPS bounds...")
        requests_per_second = enforce_min_max_rps(requests_per_second, min_rps, max_rps)
    
    # Store final scaled RPS data
    scaled_rps_data = {}
    for second in sorted(requests_per_second.keys()):
        scaled_rps_data[second] = len(requests_per_second[second])
    
    # Generate plot
    plot_name = output_file.rsplit('.', 1)[0] + '_rps_comparison.png'
    plot_rps_comparison(original_rps_data, scaled_rps_data, plot_name)
    
    # Write final output with adjusted timestamps
    print("\nWriting output file...")
    with open(output_file, 'w') as fout:
        for second in sorted(requests_per_second.keys()):
            entries = requests_per_second[second]
            for i, data in enumerate(entries):
                # Distribute entries evenly within the second
                new_timestamp = (second * 1000) + int((i / max(1, len(entries))) * 1000)
                data['timestamp'] = new_timestamp
                fout.write(json.dumps(data) + '\n')
    
    print(f"\nProcessing complete. Output written to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Scale RPS in JSONL file')
    parser.add_argument('--input_file', required=True, help='Input JSONL file path')
    parser.add_argument('--output_file', required=True, help='Output JSONL file path')
    parser.add_argument('--target_avg_rps', type=float, help='Target average RPS (optional)')
    parser.add_argument('--min_rps', type=float, help='Minimum RPS allowed (optional)')
    parser.add_argument('--max_rps', type=float, help='Maximum RPS allowed (optional)')
    
    args = parser.parse_args()
    
    try:
        process_jsonl_rps(
            args.input_file,
            args.output_file,
            args.target_avg_rps,
            args.min_rps,
            args.max_rps
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()