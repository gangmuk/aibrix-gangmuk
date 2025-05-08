from kubernetes import client, config
import json
import threading
from typing import List, Any, Dict
import time
import os
import subprocess
import random
import logging
import traceback
import datetime
import pytz

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_workload(input_path: str) -> List[Any]:
    load_struct = None
    if input_path.endswith(".jsonl"):
        with open(input_path, "r") as file:
            load_struct = [json.loads(line) for line in file]
    else:
        with open(input_path, "r") as file:
            load_struct = json.load(file)
    return load_struct

# Function to wrap the prompt into OpenAI's chat completion message format.
def prepare_prompt(prompt: str, 
                   lock: threading.Lock,
                   session_id: str = None, 
                   history: Dict = None) -> List[Dict]:
    """
    Wrap the prompt into OpenAI's chat completion message format.

    :param prompt: The user prompt to be converted.
    :return: A list containing chat completion messages.
    """
    if session_id is not None:
        with lock:
            past_history = history.get(session_id, [])
            user_message = {"role": "user", "content": f"{prompt}"}
            past_history.append(user_message) 
            history[session_id] = past_history
            return past_history
    else:    
        user_message = {"role": "user", "content": prompt}
        return [user_message]
    
def update_response(response: str, 
                    lock: threading.Lock,
                    session_id: str = None, 
                    history: Dict = None):
    """
    Wrap the prompt into OpenAI's chat completion message format.

    :param prompt: The user prompt to be converted.
    :return: A list containing chat completion messages.
    """
    if session_id is not None:
        with lock:
            past_history = history.get(session_id, [])
            assistant_message = {"role": "assistant", "content": f"{response}"}
            past_history.append(assistant_message) 


######################################################################################


def run_command(command, required=True, print_error=True, nonblock=False):
    """Run shell command and return its output or process handle.

    Args:
        command (str): The shell command to execute.
        required (bool): If True, the function will assert on failure.
        print_error (bool): If True, errors will be printed.
        nonblock (bool): If True, run the command non-blocking.

    Returns:
        tuple: 
            - If nonblock is False: (True, output) on success or (False, error) on failure.
            - If nonblock is True: (True, process) on success or (False, error) on failure.
    """
    try:
        if nonblock:
            # Start the process without waiting for it to complete
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True, process
        else:
            # Run the command and wait for it to complete
            output = subprocess.check_output(
                command,
                shell=True,
                stderr=subprocess.STDOUT,
                text=True
            )
            return True, output.strip()
    except subprocess.CalledProcessError as e:
        if print_error:
            print(f"ERROR command: {command}")
            print(f"ERROR output: {e.output.strip()}")
        if required:
            print("Exiting due to required command failure...")
            raise  # Instead of assert False, it's better to raise an exception
        else:
            return False, e.output.strip()


def check_deployment_ready_kubernetes(deployment_name, namespace):
    """
    Checks if all pods of a deployment and all their containers are in a ready state using the Kubernetes Python client.

    Args:
        deployment_name (str): The name of the deployment.
        namespace (str): The namespace where the deployment is located.

    Returns:
        bool: True if all pods and their containers are ready, False otherwise (will keep checking).
    """
    try:
        # Load Kubernetes configuration (assuming you have a valid kubeconfig file)
        config.load_kube_config()
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        
        # Add a max retry counter to prevent infinite loops
        max_retries = 60  # 1 minute with 1-second intervals
        retry_count = 0

        while retry_count < max_retries:
            try:
                deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
                selector = deployment.spec.selector.match_labels

                if not selector:
                    print(f"No selector found for deployment '{deployment_name}'. Cannot find associated pods. Retrying in 1 second...")
                    time.sleep(1)
                    retry_count += 1
                    continue

                label_selector = ",".join([f"{k}={v}" for k, v in selector.items()])
                pod_list = core_v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
                pods = pod_list.items

                if not pods:
                    print(f"No pods found for deployment '{deployment_name}' in namespace '{namespace}' with selector '{label_selector}'. Retrying in 1 second...")
                    time.sleep(1)
                    retry_count += 1
                    continue

                all_ready = True
                for pod in pods:
                    pod_name = pod.metadata.name
                    conditions = pod.status.conditions or []
                    ready_condition = next(
                        (c for c in conditions if c.type == "Ready"), None
                    )

                    if not ready_condition or ready_condition.status != "True":
                        print(f"Pod '{pod_name}' is not ready. Retrying in 1 second...")
                        all_ready = False
                        break

                    container_statuses = pod.status.container_statuses or []
                    for container_status in container_statuses:
                        if not container_status.ready:
                            container_name = container_status.name
                            print(f"Container '{container_name}' in pod '{pod_name}' is not ready. Retrying in 1 second...")
                            all_ready = False
                            break
                    if not all_ready:
                        break

                if all_ready:
                    print(f"Deployment '{deployment_name}' is ready!")
                    return True

                print(f"Deployment '{deployment_name}' is not ready yet.")
                time.sleep(1)
                retry_count += 1
                
            except client.ApiException as e:
                if e.status == 404:
                    print(f"Deployment '{deployment_name}' not found in namespace '{namespace}'. Please check the name and namespace.")
                    assert False
                else:
                    print(f"Kubernetes API exception: {e}")
                    print("Retrying in 1 second...")
                    time.sleep(1)
                    retry_count += 1
                    
        print(f"Max retries ({max_retries}) reached. Deployment '{deployment_name}' is not ready.")
        return False

    except config.ConfigException as e:
        print(f"Kubernetes configuration error: {e}")
        print("Please ensure your kubeconfig file is properly configured.")
        return False  # Exit if configuration is invalid
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False



def restart_deploy(deployment_name, namespace):
    run_command(f"kubectl rollout restart deploy {deployment_name} -n {namespace}")


def collect_k8s_logs(namespace, deployment_name, output_dir, keyword=None):
    try:
        logger.info(f"Collecting logs from {deployment_name} in namespace {namespace}")
        
        # Get the pod name directly using the deployment name (more reliable than labels)
        cmd = ["kubectl", "get", "pods", "-n", namespace, "-l", "app=gateway-plugins", "-o", "jsonpath={.items[0].metadata.name}"]
        logger.info(f"Executing command: {' '.join(cmd)}")
        pod_name = subprocess.check_output(cmd)
        pod_name = pod_name.decode('utf-8').strip()
        
        if not pod_name:
            logger.error(f"No pod found for deployment {deployment_name} in namespace {namespace}")
            # Fallback: try to get the pod name directly using the deployment name
            logger.info("Trying fallback method to find pod...")
            cmd = ["kubectl", "get", "pods", "-n", namespace, "--selector=app=gateway-plugins", "-o", "name"]
            logger.info(f"Executing command: {' '.join(cmd)}")
            output = subprocess.check_output(cmd)
            pods = output.decode('utf-8').strip().split('\n')
            if not pods or not pods[0]:
                logger.error("Fallback method also failed to find any pods")
                return False
            # Take the first pod name and remove the "pod/" prefix
            pod_name = pods[0].replace("pod/", "")
        
        logger.info(f"Found pod: {pod_name}")
        
        # Get logs from the pod
        cmd = ["kubectl", "logs", "-n", namespace, pod_name]
        logger.info(f"Executing command: {' '.join(cmd)}")
        logs = subprocess.check_output(cmd)
        logs_str = logs.decode('utf-8')
        
        # Process logs based on keyword filter
        if keyword:
            logger.info(f"Filtering logs for lines containing keyword: '{keyword}'")
            filtered_lines = []
            total_lines = 0
            filtered_count = 0
            
            for line in logs_str.splitlines():
                total_lines += 1
                if keyword in line:
                    filtered_lines.append(line)
                    filtered_count += 1
            
            filtered_output_content = '\n'.join(filtered_lines)
            logger.info(f"Filtered {filtered_count} lines containing keyword from {total_lines} total lines")
        
        # Write filtered logs to file
        filtered_log_output_file = f"{output_dir}/filtered_gateway_plugins.log.csv"
        with open(filtered_log_output_file, 'w', encoding='utf-8') as f:
            f.write(filtered_output_content)
        all_log_output_file = f"{output_dir}/all_gateway_plugins.log.txt"
        with open(all_log_output_file, 'w', encoding='utf-8') as f:
            f.write(logs_str)
        
        output_size = len(filtered_output_content) / 1024  # Size in KB
        logger.info(f"Unfiltered Logs saved to {all_log_output_file} ({output_size:.2f} KB)")
        logger.info(f"Filtered Logs saved to {filtered_log_output_file} ({output_size:.2f} KB)")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing kubectl command: {e}")
        logger.error(f"Command output: {e.output.decode('utf-8') if hasattr(e, 'output') else 'No output'}")
        return False
    except Exception as e:
        logger.error(f"Error collecting logs: {e}")
        logger.error(traceback.format_exc())
        return False


def get_current_pdt_time():
    """Get the current time in PDT timezone"""
    pdt_tz = pytz.timezone('America/Los_Angeles')
    utc_now = datetime.datetime.utcnow()
    utc_now = utc_now.replace(tzinfo=pytz.utc)
    pdt_now = utc_now.astimezone(pdt_tz)
    return pdt_now.strftime("%Y%m%d_%H%M%S")