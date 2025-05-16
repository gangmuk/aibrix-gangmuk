#!/usr/bin/env python3
import os
import subprocess
import argparse

def scp_filtered_logs(remote_host, base_path, local_dir='.'):
    """
    SCP all files named 'filtered-aibrix-gateway-plugins.log.csv' from subdirectories
    under the specified path on the remote host.
    
    Args:
        remote_host (str): Remote host in the format 'user@hostname'
        base_path (str): Base path on the remote host
        local_dir (str): Local directory to save files to
    """
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Command to find all relevant files and their directories
    find_cmd = f"ssh {remote_host} 'find {base_path} -name \"filtered-aibrix-gateway-plugins.log.csv\" -type f'"
    
    try:
        # Execute the find command and get the results
        result = subprocess.run(find_cmd, shell=True, check=True, text=True, capture_output=True)
        file_paths = result.stdout.strip().split('\n')
        
        # Filter out empty lines
        file_paths = [path for path in file_paths if path]
        
        if not file_paths:
            print(f"No 'filtered-aibrix-gateway-plugins.log.csv' files found under {base_path}")
            return
            
        print(f"Found {len(file_paths)} files to copy")
        
        # Copy each file with the subdirectory name as prefix
        for remote_path in file_paths:
            # Get the last subdirectory name
            routing_strategy = os.path.basename(os.path.dirname(remote_path))
            
            workloadname = base_path.split('/')[-1]
            # Create the new local filename
            dir_ = f"{local_dir}/{workloadname}/{routing_strategy}"
            print(f"Creating directory: {dir_}")
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            local_path = f"{dir_}/filtered-aibrix-gateway-plugins.log.csv"
            # SCP the file
            scp_cmd = f"scp {remote_host}:{remote_path} {local_path}"
            print(f"Copying: {remote_path} -> {local_path}")
            
            subprocess.run(scp_cmd, shell=True, check=True)
            
        print("All files copied successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy filtered log files from remote host')
    parser.add_argument('remote_host', help='Remote host in format user@hostname')
    parser.add_argument('base_path', help='Base path on remote host')
    parser.add_argument('--local-dir', default='.', help='Local directory to save files to')
    
    args = parser.parse_args()
    
    scp_filtered_logs(args.remote_host, args.base_path, args.local_dir)