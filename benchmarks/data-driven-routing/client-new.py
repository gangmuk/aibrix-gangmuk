import argparse
import logging
import time
import asyncio
import openai
import json
import io
import traceback
import threading
from queue import Queue
import csv
import os
import aiohttp
from typing import List
from utils import (load_workload, prepare_prompt, update_response)
import httpx
from openai._base_client import BaseClient


thread_pool_size = 10
QUEUE_SIZE = thread_pool_size * 2
logging.basicConfig(level=logging.INFO)
task_queue = Queue(maxsize=QUEUE_SIZE)
session_history = {}
lock = threading.Lock()
csv_file_name = 'output.csv'

class HeaderCaptureTransport(httpx.AsyncHTTPTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_headers = {}
        
    async def handle_async_request(self, request):
        response = await super().handle_async_request(request)
        self.captured_headers = dict(response.headers)
        return response

# Create a function to patch the OpenAI client
def patch_openai_client(client):
    """Patch the OpenAI client to capture headers from responses."""
    transport = HeaderCaptureTransport()
    # Access the internal httpx client and modify its transport
    if hasattr(client, "_client"):
        client._client._transport = transport
    elif hasattr(client, "client"):
        client.client._transport = transport
    
    return transport

def worker(thread_idx, client, model, send_request_func, output_file):
    """Worker function to run an asyncio event loop in a separate thread."""
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    while True:
        task = task_queue.get()
        if task is None:  # Stop signal
            logging.warning(f"Worker {thread_idx} exit.")
            break
        else:
            loop.run_until_complete(send_request_func(client, model, *task))
        task_queue.task_done()


def start_worker_threads(thread_idx, client, model, send_request_func, output_file):
    """Start multiple threads, each running an event loop for handling tasks."""
    thread = threading.Thread(target=worker, args=(thread_idx, client, model, send_request_func, output_file), daemon=True)
    thread.start()
    return thread


def write_result_to_files(result_data, jsonl_file, csv_file=None):
    # Write to JSONL file
    jsonl_file.write(json.dumps(result_data) + "\n")
    jsonl_file.flush()  # Ensure data is written immediately
    
    # Skip CSV writing if no file specified
    if csv_file is None:
        return
        
    # Handle CSV writing - check if csv_file is a string (path) or file object
    csv_file_obj = None
    close_csv_file = False
    
    try:
        if isinstance(csv_file, str):
            # It's a file path, need to open the file
            file_exists = os.path.exists(csv_file)
            csv_file_obj = open(csv_file, 'a' if file_exists else 'w')
            close_csv_file = True  # We need to close it after writing
            is_new_file = not file_exists or os.path.getsize(csv_file) == 0
        else:
            # It's already a file object
            csv_file_obj = csv_file
            is_new_file = csv_file_obj.tell() == 0
            
        # Write to CSV file
        csv_writer = csv.DictWriter(csv_file_obj, fieldnames=result_data.keys())
        
        if is_new_file:
            csv_writer.writeheader()
        
        # Convert complex JSON structures to strings for CSV compatibility
        csv_row = {}
        for key, value in result_data.items():
            if isinstance(value, (dict, list)):
                csv_row[key] = json.dumps(value)
            else:
                csv_row[key] = value
                
        csv_writer.writerow(csv_row)
        csv_file_obj.flush()  # Ensure data is written immediately
        
    finally:
        # Close the file if we opened it
        if close_csv_file and csv_file_obj:
            csv_file_obj.close()

# Shared helper functions

def parse_json_header(headers, header_name):
    """Parse a JSON header safely, returning None if parsing fails."""
    if header_name in headers:
        try:
            return json.loads(headers.get(header_name))
        except json.JSONDecodeError:
            logging.warning(f"Could not parse {header_name} header")
    return "Not Found"

def extract_headers_data(headers):
    """Extract and parse all relevant headers from the response."""
    # Basic headers
    selected_pod_ip = headers.get('target-pod', 'Not Found')
    selected_pod_name = headers.get('target-pod-name', 'Not Found')
    
    # Log missing important headers
    if not selected_pod_name:
        logging.error("target-pod-name header not found in response.")
    if not selected_pod_ip:
        logging.error("target-pod header not found in response.")
    
    # Timing headers
    gateway_side_ttft = float(headers.get('x-timing-ttft-ms', -1))
    gateway_side_tpot = float(headers.get('x-timing-tpot-ms', -1))
    gateway_side_e2e_latency = float(headers.get('x-timing-e2e-ms', -1))
    kv_cache_hit_ratio = float(headers.get('x-kvcache-hit-ratio', -1))
    
    # Complex JSON headers
    all_pods_kv_cache_hit_ratio = parse_json_header(headers, 'x-kvcache-hit-ratio-all')
    all_pods_num_inflight_requests = parse_json_header(headers, 'x-num-inflight-requests-all')
    vllm_gpu_kv_cache_usage = parse_json_header(headers, 'x-vllm-gpu-kvcache-usage')
    vllm_cpu_kv_cache_usage = parse_json_header(headers, 'x-vllm-cpu-kvcache-usage')
    vllm_num_running_requests = parse_json_header(headers, 'x-vllm-num-running-requests')
    vllm_num_waiting_requests = parse_json_header(headers, 'x-vllm-num-waiting-requests')
    captured_headers_data = {
        "selected_pod_ip": selected_pod_ip,
        "selected_pod_name": selected_pod_name,
        "gateway_side_ttft": gateway_side_ttft,
        "gateway_side_tpot": gateway_side_tpot,
        "gateway_side_e2e_latency": gateway_side_e2e_latency,
        "kv_cache_hit_ratio": kv_cache_hit_ratio,
        "kv_cache_hit_ratio": all_pods_kv_cache_hit_ratio,
        "num_inflight_requests": all_pods_num_inflight_requests,
        "vllm_gpu_kv_cache_usage": vllm_gpu_kv_cache_usage,
        "vllm_cpu_kv_cache_usage": vllm_cpu_kv_cache_usage,
        "vllm_num_running_requests": vllm_num_running_requests,
        "vllm_num_waiting_requests": vllm_num_waiting_requests
    }
    logging.debug(f"Captured headers: {captured_headers_data}")
    return captured_headers_data

def calculate_slo_metrics(prompt_tokens, output_tokens, gateway_side_ttft, gateway_side_tpot, gateway_side_e2e_latency):
    """Calculate SLO metrics based on token counts and latencies."""
    per_token_ttft_slo_in_ms = 1000
    per_token_tpot_slo_in_ms = 40
    
    ttft_slo_in_ms = per_token_ttft_slo_in_ms * prompt_tokens
    tpot_slo_in_ms = per_token_tpot_slo_in_ms * output_tokens
    e2e_slo_in_ms = ttft_slo_in_ms + tpot_slo_in_ms
    
    e2e_slo_satisfied = gateway_side_e2e_latency <= e2e_slo_in_ms
    ttft_slo_satisfied = gateway_side_ttft <= ttft_slo_in_ms
    tpot_slo_satisfied = gateway_side_tpot <= tpot_slo_in_ms
    
    return {
        "e2e_slo_in_ms": e2e_slo_in_ms,
        "ttft_slo_in_ms": ttft_slo_in_ms,
        "tpot_slo_in_ms": tpot_slo_in_ms,
        "e2e_slo_satisfied": e2e_slo_satisfied,
        "ttft_slo_satisfied": ttft_slo_satisfied,
        "tpot_slo_satisfied": tpot_slo_satisfied
    }

def create_success_result(request_id, start_time, response_time, client_side_ttft, client_side_tpot, prompt_tokens, output_tokens, total_tokens, 
                         headers_data, prompt, session_id=None):
    """Create a result dictionary for successful requests."""
    # Calculate client-side latency
    client_side_e2e_latency_in_ms = (response_time - start_time) * 1000
    throughput = output_tokens / (client_side_e2e_latency_in_ms / 1000) if output_tokens > 0 else 0
    
    # Calculate SLO metrics
    slo_metrics = calculate_slo_metrics(
        prompt_tokens, 
        output_tokens, 
        headers_data["gateway_side_ttft"], 
        headers_data["gateway_side_tpot"], 
        headers_data["gateway_side_e2e_latency"]
    )
    
    # Combine all data into a result dictionary
    result = {
        "request_id": request_id,
        "status": "success",
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "selected_pod_ip": headers_data["selected_pod_ip"],
        "selected_pod_name": headers_data["selected_pod_name"],
        "gpu_model": "NVIDIA-L20",
        "kv_cache_hit_ratio": headers_data["kv_cache_hit_ratio"],
        "num_inflight_requests": headers_data["num_inflight_requests"],
        "vllm_gpu_kv_cache_usage": headers_data["vllm_gpu_kv_cache_usage"],
        "vllm_cpu_kv_cache_usage": headers_data["vllm_cpu_kv_cache_usage"],
        "vllm_num_running_requests": headers_data["vllm_num_running_requests"],
        "vllm_num_waiting_requests": headers_data["vllm_num_waiting_requests"],
        # "input": prompt,
        "client_side_token_per_second": f"{throughput:.2f}",
        "client_side_start_time": f"{start_time:.2f}",
        "client_side_end_time": f"{response_time:.2f}",
        "client_side_e2e_latency_in_ms": f"{client_side_e2e_latency_in_ms:.4f}",
        "client_side_ttft": client_side_ttft,
        "client_side_tpot": client_side_tpot,
        "gateway_side_ttft": headers_data["gateway_side_ttft"],
        "gateway_side_tpot": headers_data["gateway_side_tpot"],
        "gateway_side_e2e_latency": headers_data["gateway_side_e2e_latency"],
        "e2e_slo_in_ms": slo_metrics["e2e_slo_in_ms"],
        "ttft_slo_in_ms": slo_metrics["ttft_slo_in_ms"],
        "tpot_slo_in_ms": slo_metrics["tpot_slo_in_ms"],
        "e2e_slo_satisfied": slo_metrics["e2e_slo_satisfied"],
        "ttft_slo_satisfied": slo_metrics["ttft_slo_satisfied"],
        "tpot_slo_satisfied": slo_metrics["tpot_slo_satisfied"],
        "error_type": None,
        "error_message": None,
        "error_traceback": None,
    }
    if session_id:
        result["session_id"] = session_id
    return result

def create_error_result(request_id, start_time, error_time, e, prompt, selected_pod_ip="", selected_pod_name="", session_id=None):
    """Create a result dictionary for failed requests."""
    error_type = type(e).__name__
    client_side_e2e_latency_in_ms = (error_time - start_time) * 1000
    
    result = {
        "request_id": request_id,
        "status": "error",
        "prompt_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "client_side_token_per_second": None,
        "client_side_start_time": f"{start_time:.2f}",
        "client_side_end_time": f"{error_time:.2f}",
        "client_side_e2e_latency_in_ms": f"{client_side_e2e_latency_in_ms:.4f}",
        "client_side_ttft": None,
        "client_side_tpot": None,
        "gateway_side_ttft": None,
        "gateway_side_tpot": None,
        "gateway_side_e2e_latency": None,
        "selected_pod_ip": selected_pod_ip,
        "selected_pod_name": selected_pod_name,
        "gpu_model": None,
        "kv_cache_hit_ratio": None,
        "num_inflight_requests": None,
        "vllm_gpu_kv_cache_usage": None,
        "vllm_cpu_kv_cache_usage": None,
        "vllm_num_running_requests": None,
        "vllm_num_waiting_requests": None,
        "e2e_slo_in_ms": None,
        "ttft_slo_in_ms": None,
        "tpot_slo_in_ms": None,
        "e2e_slo_satisfied": None,
        "ttft_slo_satisfied": None,
        "tpot_slo_satisfied": None,
        "error_type": error_type,
        "error_message": str(e),
        "error_traceback": traceback.format_exc(),
        # "input": prompt
    }
    
    # Add session_id if provided
    if session_id:
        result["session_id"] = session_id
    
    return result

# Refactored functions using shared components

async def send_request_batch_for_mock_app_format(client: openai.AsyncOpenAI,
                             model: str,
                             endpoint: str,
                             prompt: str,
                             output_file: io.TextIOWrapper,
                             request_id: int,
                             max_tokens: int,
                             routing_strategy: str,
                             target_time: int = None):
    
    start_time = asyncio.get_event_loop().time()
    selected_pod_ip = ""
    selected_pod_name = ""
    client_side_ttft = None
    client_side_tpot = None
    
    # Initialize CSV file if needed
    if request_id == 0:
        csv_file = open(csv_file_name, 'w', encoding='utf-8')
    
    try:
        # Handle target time if specified
        if target_time is not None:
            cur_time = time.time()
            if target_time > cur_time:
                await asyncio.sleep(target_time - cur_time)
        
        logging.info(f"Request {request_id}: Starting sending request to {endpoint}")
        
        # Extract the content from the message format
        prompt_content = prompt[0]["content"] if isinstance(prompt, list) and len(prompt) > 0 and "content" in prompt[0] else prompt
        
        # Use aiohttp for direct HTTP request
        headers = {
            "Content-Type": "application/json",
            "model": model,
            "routing-strategy": routing_strategy
        }
        
        # Add Authorization if API key is provided
        if hasattr(client, "api_key") and client.api_key:
            headers["Authorization"] = f"Bearer {client.api_key}"
            
        # Construct payload
        max_tokens = 2048 if max_tokens is None else max_tokens
        payload = {
            "model": model,
            "prompt": prompt_content,
            "temperature": 0,
            "max_tokens": max_tokens
        }
        
        # Send request
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{endpoint}/v1/chat/completions", 
                                  headers=headers, 
                                  json=payload) as response:
                
                if response.status != 200:
                    raise Exception(f"Request failed with status {response.status}: {await response.text()}")
                
                # Log response headers
                all_headers = dict(response.headers)
                logging.info(f"Request {request_id}: Response headers:")
                for header_name, header_value in all_headers.items():
                    logging.info(f"  {header_name}: {header_value}")
                
                # Parse response
                response_data = await response.json()
                headers_data = extract_headers_data(response.headers)
                
                # Calculate response time
                response_time = asyncio.get_event_loop().time()
                
                # Extract token counts from response
                prompt_tokens = int(response_data.get("usage", {}).get("prompt_tokens", 0))
                output_tokens = int(response_data.get("usage", {}).get("completion_tokens", 0))
                total_tokens = int(response_data.get("usage", {}).get("total_tokens", 0))
                
                # Create success result
                result = create_success_result(
                    request_id=request_id,
                    start_time=start_time,
                    response_time=response_time,
                    client_side_ttft=client_side_ttft,
                    client_side_tpot=client_side_tpot,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    headers_data=headers_data,
                    prompt=prompt
                )
        
        # Log success
        logging.info(f"Request {request_id}: Completed successfully. Input tokens: {result['prompt_tokens']}, "
                     f"Output tokens: {result['output_tokens']}, Total tokens: {result['total_tokens']}, "
                     f"client_side_e2e_latency_in_ms: {float(result['client_side_e2e_latency_in_ms']):.2f}ms")
        
        # Write results to files
        write_result_to_files(result, output_file, csv_file_name)
        
        return result

    except Exception as e:
        error_time = asyncio.get_event_loop().time()
        
        # Create error result
        error_result = create_error_result(
            request_id=request_id,
            start_time=start_time,
            error_time=error_time,
            e=e,
            prompt=prompt,
            selected_pod_ip=selected_pod_ip,
            selected_pod_name=selected_pod_name
        )
        
        # Log error
        logging.error(f"Request {request_id}: Error ({error_result['error_type']}): {error_result['error_message']}")
        logging.error(f"traceback.format_exc(): {error_result['error_traceback']}")
        
        # Write error results to files
        write_result_to_files(error_result, output_file, csv_file_name)
        
        return error_result
    
async def send_request_streaming(client: openai.AsyncOpenAI,
                             model: str,
                             prompt: str,
                             output_file: io.TextIOWrapper,
                             request_id: int,
                             session_id: str,
                             target_time: int,
                             max_tokens: int = 2048,
                             temperature: float = 0.0,
                             routing_strategy: str = None):

    start_time = asyncio.get_event_loop().time()
    first_response_time = None
    selected_pod_ip = ""
    selected_pod_name = ""
    client_side_ttft = -1
    client_side_tpot = -1
    try:
        # Handle target time if specified
        if target_time is not None:
            cur_time = time.time()
            logging.warning(f"send_request_streaming: Prepare to launch task after {target_time - cur_time}")
            if target_time > cur_time:
                await asyncio.sleep(target_time - cur_time)
        
        logging.info(f"Request {request_id}: Starting streaming request")
        
        # Set additional headers if needed
        extra_headers = {}
        if routing_strategy:
            extra_headers["routing-strategy"] = routing_strategy
        
        # Send streaming request
        transport = patch_openai_client(client)

        response_stream = await client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_headers=extra_headers
        )
        
        headers_data = extract_headers_data(transport.captured_headers)
        
        text_chunks = []
        prompt_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        try:
            async for chunk in response_stream:
                if chunk.choices:
                    if chunk.choices[0].delta.content is not None:
                        if first_response_time is None:
                            first_response_time = asyncio.get_event_loop().time()
                        output_text = chunk.choices[0].delta.content
                        text_chunks.append(output_text)
                
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    # For OpenAI, we expect to get complete usage stats, not partial ones to accumulate
                    if chunk.usage.prompt_tokens is not None:
                        prompt_tokens = chunk.usage.prompt_tokens
                    if chunk.usage.completion_tokens is not None:
                        output_tokens = chunk.usage.completion_tokens
                    if chunk.usage.total_tokens is not None:
                        total_tokens = chunk.usage.total_tokens
        
        except Exception as stream_error:
            # Handle errors during streaming
            logging.error(f"Request {request_id}: Stream interrupted: {type(stream_error).__name__}: {str(stream_error)}")
            raise stream_error
        
        # Combine text chunks to get full response
        response_text = "".join(text_chunks)
        response_time = asyncio.get_event_loop().time()
        
        # Update session history if needed
        if 'update_response' in globals() and 'lock' in globals() and 'session_history' in globals():
            update_response(response=response_text, lock=lock, session_id=session_id, history=session_history)
        
        # Calculate streaming metrics
        client_side_ttft = (first_response_time - start_time) * 1000 if first_response_time else None
        client_side_tpot = ((response_time - first_response_time) * 1000 / output_tokens) if first_response_time and output_tokens > 0 else None
        
        # Create success result using the shared function
        result = create_success_result(
            request_id=request_id,
            start_time=start_time,
            response_time=response_time,
            client_side_ttft=client_side_ttft,
            client_side_tpot=client_side_tpot,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            headers_data=headers_data,
            prompt=prompt,
            session_id=session_id
        )
        
        # Log success
        logging.info(f"Request {request_id}: Completed successfully. Input tokens: {prompt_tokens}, "
                    f"Output tokens: {output_tokens}, Total tokens: {total_tokens}, "
                    f"client_side_e2e_latency_in_ms: {float(result['client_side_e2e_latency_in_ms']):.2f}ms")
        
        write_result_to_files(result, output_file, csv_file_name)
        return result
    
    except Exception as e:
        error_time = asyncio.get_event_loop().time()
        
        # Create error result using shared function - exactly like send_request_batch does
        error_result = create_error_result(
            request_id=request_id,
            start_time=start_time,
            error_time=error_time,
            e=e,
            prompt=prompt,
            selected_pod_ip=selected_pod_ip,
            selected_pod_name=selected_pod_name,
            session_id=session_id
        )
        
        # Log error
        logging.error(f"Request {request_id}: Error ({error_result['error_type']}): {error_result['error_message']}")
        logging.error(f"traceback.format_exc(): {error_result['error_traceback']}")
        
        write_result_to_files(error_result, output_file, csv_file_name)
        
        return error_result

async def benchmark_streaming(api_key: str,
                              endpoint: str,
                              max_retries: int,
                              timeout: float,
                              routing_strategy: str,
                              load_struct: List,
                              output_file: io.TextIOWrapper,
                              model: str, 
                              max_tokens: int, 
                              temperature: float):
    request_id = 0
    base_time = time.time()
    num_requests = 0
    threads = []
    for thread_idx in range(0, thread_pool_size):
        client = create_client(api_key, endpoint, max_retries, timeout, routing_strategy)
        threads.append(start_worker_threads(thread_idx, client, model, send_request_streaming, output_file))
    for requests_dict in load_struct:
        ts = int(requests_dict["timestamp"])
        requests = requests_dict["requests"]
        target_time = base_time + ts / 1000.0
        formatted_prompts = [prepare_prompt(prompt = request["prompt"], lock = lock, session_id = request.get("session_id", None), history = session_history) for request in requests]
        for i in range(len(requests)):
            session_id = requests[i].get("session_id", None)
            task_queue.put((formatted_prompts[i], output_file, request_id, session_id, target_time, max_tokens, temperature))
            request_id += 1
        num_requests += len(requests)
    task_queue.join()
    # Stop all worker threads
    logging.warning("Producer completed ...")
    for _ in threads:
        task_queue.put(None)

    for thread in threads:
        thread.join()
        logging.warning(f"Worker thread {thread} completed ...")
    logging.warning(f"All {num_requests} requests completed for deployment.")


async def send_request_batch(client: openai.AsyncOpenAI,
                             model: str,
                             endpoint: str,
                             prompt: str,
                             output_file: io.TextIOWrapper,
                             request_id: int,
                             session_id: str, 
                             target_time: int,
                             max_tokens: int = 2048,
                             temperature: float = 0.0,
                             routing_strategy: str = None):
    
    start_time = asyncio.get_event_loop().time()
    selected_pod_ip = ""
    selected_pod_name = ""
    client_side_ttft = -1
    client_side_tpot = -1
    try:
        # Handle target time if specified
        if target_time is not None:
            cur_time = time.time()
            logging.warning(f"send_request_batch: Prepare to launch task after {target_time - cur_time}")
            if target_time > cur_time:
                await asyncio.sleep(target_time - cur_time)
        
        logging.info(f"Request {request_id}: Starting sending request to {endpoint}")
        
        # Set additional headers if needed
        extra_headers = {}
        if routing_strategy:
            extra_headers["routing-strategy"] = routing_strategy

        transport = patch_openai_client(client)

        # Send request using the OpenAI client
        response = await client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_headers=extra_headers
        )
        
        headers_data = extract_headers_data(transport.captured_headers)

        # Extract response time and token counts
        response_time = asyncio.get_event_loop().time()
        prompt_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        output_text = response.choices[0].message.content
        
        # Update session history if needed
        if 'update_response' in globals() and 'lock' in globals() and 'session_history' in globals():
            update_response(response=output_text, lock=lock, session_id=session_id, history=session_history)

        # Create success result
        result = create_success_result(
            request_id=request_id,
            start_time=start_time,
            response_time=response_time,
            client_side_ttft=client_side_ttft,
            client_side_tpot=client_side_tpot,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            headers_data=headers_data,
            prompt=prompt,
            session_id=session_id
        )
        
        # Log success
        logging.info(f"Request {request_id}: Completed successfully. Input tokens: {prompt_tokens}, "
                     f"Output tokens: {output_tokens}, Total tokens: {total_tokens}, "
                     f"client_side_e2e_latency_in_ms: {float(result['client_side_e2e_latency_in_ms']):.2f}ms")

        write_result_to_files(result, output_file, csv_file_name)
        return result

    except Exception as e:
        error_time = asyncio.get_event_loop().time()
        
        # Create error result
        error_result = create_error_result(
            request_id=request_id,
            start_time=start_time,
            error_time=error_time,
            e=e,
            prompt=prompt,
            selected_pod_ip=selected_pod_ip,
            selected_pod_name=selected_pod_name,
            session_id=session_id
        )
        
        # Log error
        logging.error(f"Request {request_id}: Error ({error_result['error_type']}): {error_result['error_message']}")
        logging.error(f"traceback.format_exc(): {error_result['error_traceback']}")
        
        write_result_to_files(error_result, output_file, csv_file_name)
        
        return error_result


async def benchmark_batch(api_key: str,
                          endpoint: str,
                          max_retries: int,
                          timeout: float,
                          routing_strategy: str,
                          load_struct: List,
                          output_file: io.TextIOWrapper,
                          model: str,
                          max_tokens: int = None,
                          temperature: float = 0.0):
    request_id = 0
    base_time = time.time()
    num_requests = 0
    threads = []
    
    for thread_idx in range(0, thread_pool_size):
        client = create_client(api_key, endpoint, max_retries, timeout, routing_strategy)
        threads.append(start_worker_threads(thread_idx, client, model, send_request_batch, output_file))
        # threads.append(start_worker_threads(thread_idx, client, model, send_request_batch_for_mock_app_format, output_file))
    
    for requests_dict in load_struct:
        ts = int(requests_dict["timestamp"])
        requests = requests_dict["requests"]
        target_time = base_time + ts / 1000.0
        formatted_prompts = [prepare_prompt(prompt = request["prompt"], lock = lock, session_id = request.get("session_id", None), history = session_history) for request in requests]
        
        for i in range(len(requests)):
            session_id = requests[i].get("session_id", None)
            task_queue.put((endpoint, formatted_prompts[i], output_file, request_id, session_id, target_time, max_tokens, temperature, routing_strategy))
            request_id += 1
        
        num_requests += len(requests)
    
    task_queue.join()
    # Stop all worker threads
    for _ in threads:
        task_queue.put(None)

    for thread in threads:
        thread.join()
        logging.warning(f"Worker thread {thread} completed ...")
    logging.warning(f"All {num_requests} requests completed for deployment.")


def create_client(api_key: str,
                  endpoint: str,
                  max_retries: int,
                  timeout: float,
                  routing_strategy: str,
                  ):
    if api_key is None:
        client = openai.AsyncOpenAI(
            base_url=endpoint + "/v1",
            max_retries=max_retries,
            timeout=timeout,
        )
    else:
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint + "/v1",
            max_retries=max_retries,
            timeout=timeout,
        )
    if routing_strategy is not None:
        client = client.with_options(
            default_headers={"routing-strategy": routing_strategy}
        )
    return client

def main(args):
    logging.info(f"Starting benchmark on endpoint {args.endpoint}")
    # write empty line to csv_file_name
    with open(csv_file_name, 'w', encoding='utf-8') as csv_file:
        csv_file.write("")
    with open(args.output_file_path, 'w', encoding='utf-8') as output_file:
        load_struct = load_workload(args.workload_path)
        if not args.streaming:
            logging.info("Using batch client")
            start_time = time.time()
            asyncio.run(benchmark_batch(
                api_key = args.api_key,
                endpoint = args.endpoint,
                max_retries = 0,
                timeout = 60.0,
                routing_strategy = args.routing_strategy,
                load_struct=load_struct,
                output_file=output_file,
                model=args.model,
                max_tokens = args.max_tokens,
                temperature=args.temperature,
            ))
            end_time = time.time()
            logging.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")
        else:
            logging.info("Using streaming client")
            start_time = time.time()
            asyncio.run(benchmark_streaming(
                api_key = args.api_key,
                endpoint = args.endpoint,
                max_retries = 0,
                timeout = 60.0,
                routing_strategy = args.routing_strategy,
                load_struct=load_struct,
                output_file=output_file,
                model=args.model,
                max_tokens = args.max_tokens,
                temperature=args.temperature,
            ))
            end_time = time.time()
            logging.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Workload Generator')
    parser.add_argument("--workload-path", type=str, default=None, help="File path to the workload file.")
    parser.add_argument("--model", type=str, required=True, default=None, help="Default target model (if workload does not contains target model).")
    parser.add_argument('--endpoint', type=str, required=True)
    parser.add_argument("--api-key", type=str, default=None, help="API key to the service. ")
    parser.add_argument('--output-file-path', type=str, default="output.jsonl")
    parser.add_argument("--streaming", action="store_true", help="Use streaming client.")
    parser.add_argument("--routing-strategy", type=str, required=False, default="random", help="Routing strategy to use.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for the request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the request.")
    args = parser.parse_args()
    main(args)
