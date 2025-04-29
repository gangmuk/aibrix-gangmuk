import argparse
import logging
import time
import asyncio
import openai
import json
import io
import traceback
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import csv
import os
import aiohttp
import httpx
import re
import utils


# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
session_history = {}
csv_file_name = ''

class HeaderCaptureTransport(httpx.AsyncHTTPTransport):
    """Custom transport to capture response headers"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_headers = {}
        
    async def handle_async_request(self, request):
        response = await super().handle_async_request(request)
        self.captured_headers = dict(response.headers)
        return response

async def load_workload(workload_path: str) -> List[Dict[str, Any]]:
    """Load workload file asynchronously"""
    async with aiohttp.ClientSession() as session:
        try:
            with open(workload_path, 'r', encoding='utf-8') as f:
                load_struct = []
                for line in f:
                    if line.strip():
                        load_struct.append(json.loads(line))
                return load_struct
        except Exception as e:
            logger.error(f"Error loading workload: {e}")
            raise

async def send_request_streaming(client, model, prompt, output_file, request_id, 
                                session_id, target_time, max_tokens,
                                temperature, routing_strategy, results_lock, history_lock):
    """Send a streaming request asynchronously"""
    start_time = asyncio.get_running_loop().time()
    first_response_time = None
    selected_pod_ip = ""
    selected_pod_name = ""
    client_side_ttft = -1
    client_side_tpot = -1
    scheduled_time = target_time
    actual_start_time = time.time()
    
    try:
        # If target_time is provided, wait until that time
        if target_time is not None:
            current_time = time.time()
            if current_time < target_time:
                schedule_delay = target_time - current_time
                logger.info(f"Request {request_id}: Scheduled for {time.strftime('%H:%M:%S.%f', time.localtime(target_time))[:-3]}, " 
                          f"waiting {schedule_delay:.3f}s")
                await asyncio.sleep(schedule_delay)
            
            # Record the actual start time after waiting
            actual_start_time = time.time()
            scheduling_accuracy = actual_start_time - target_time
            logger.info(f"Request {request_id}: Starting streaming request at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]}, "
                      f"scheduling accuracy: {scheduling_accuracy:.6f}s")
        else:
            logger.info(f"Request {request_id}: Starting streaming request at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]} (no scheduled time)")
        
        # Double-check prompt format
        if not isinstance(prompt, list):
            # Convert to list format for chat completions
            prompt = [{"role": "user", "content": str(prompt)}]
        else:
            assert prompt, "Prompt list should not be empty"
        
        # Ensure each item in the list has role and content
        for i, msg in enumerate(prompt):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                prompt[i] = {"role": "user", "content": str(msg)}
        
        # Format validation logging
        logger.debug(f"Request {request_id}: Formatted prompt for streaming: {prompt}")
        
        # Set additional headers if needed
        extra_headers = {}
        if routing_strategy:
            extra_headers["routing-strategy"] = routing_strategy
        
        # Patch the client to capture headers
        transport = patch_openai_client(client)

        # Send streaming request
        response_stream = await client.chat.completions.create(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_headers=extra_headers
        )
        
        # Extract headers
        headers_data = extract_headers_data(transport.captured_headers)
        # print(f"Request {request_id}, headers_data: {headers_data}")
        # Process streaming response
        text_chunks = []
        prompt_tokens = 0
        output_tokens = 0
        total_tokens = 0
        ttft_logged = False
        
        async for chunk in response_stream:
            if chunk.choices:
                if chunk.choices[0].delta.content is not None:
                    if first_response_time is None:
                        first_response_time = asyncio.get_running_loop().time()
                        first_token_time = time.time()
                        ttft = first_token_time - actual_start_time
                        if not ttft_logged:
                            logger.info(f"Request {request_id}: First token received at {time.strftime('%H:%M:%S.%f', time.localtime(first_token_time))[:-3]}, "
                                        f"TTFT: {ttft:.3f}s")
                            ttft_logged = True
                            
                    output_text = chunk.choices[0].delta.content
                    text_chunks.append(output_text)
            
            # Extract usage information if available
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                if chunk.usage.prompt_tokens is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                if chunk.usage.completion_tokens is not None:
                    output_tokens = chunk.usage.completion_tokens
                if chunk.usage.total_tokens is not None:
                    total_tokens = chunk.usage.total_tokens
        # Combine text chunks to get full response
        response_text = "".join(text_chunks)
        # print(f"Request {request_id}, response_text: {response_text}")
        response_time = asyncio.get_running_loop().time()
        completion_time = time.time()
    
        # Update session history if needed
        if session_id:
            await update_response(response_text, session_id, history_lock)
        
        # Calculate streaming metrics
        client_side_ttft = (first_response_time - start_time) * 1000 if first_response_time else None
        client_side_tpot = ((response_time - first_response_time) * 1000 / output_tokens) if first_response_time and output_tokens > 0 else None
        
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
            prompt_text=prompt,
            output_text=response_text,
            session_id=session_id
        )
        
        # Calculate total elapsed time
        total_elapsed = completion_time - actual_start_time
        tpot = (completion_time - first_token_time) if first_response_time else 0
        tpot_per_token = tpot / output_tokens if output_tokens > 0 else 0
        
        logger.info(f"Request {request_id}: Completed at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]}. "
                    f"Elapsed: {total_elapsed:.3f}s, "
                    f"Tokens: {prompt_tokens} in / {output_tokens} out, "
                    f"TPOT: {tpot:.3f}s ({tpot_per_token*1000:.2f}ms/token), "
                    f"E2E latency: {float(result['client_side_e2e_latency_in_ms']):.2f}ms")
        
        # Log scheduling information
        if scheduled_time:
            scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
            actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
            logger.info(f"Request {request_id}: Scheduling summary - "
                        f"Scheduled: {scheduled_dt}, "
                        f"Started: {actual_dt}, "
                        f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")
        
        # Write results
        await write_result_to_files(result, output_file, csv_file_name, results_lock)
        return result
    
    except Exception as e:
        error_time = asyncio.get_running_loop().time()
        completion_time = time.time()
        
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
        
        # Calculate total elapsed time
        total_elapsed = completion_time - actual_start_time
        
        logger.error(f"Request {request_id}: Streaming error at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]} "
                   f"after {total_elapsed:.3f}s: {error_result['error_type']}: {error_result['error_message']}")
        
        # Log scheduling information for errors too
        if scheduled_time:
            scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
            actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
            logger.error(f"Request {request_id}: Scheduling error summary - "
                      f"Scheduled: {scheduled_dt}, "
                      f"Started: {actual_dt}, "
                      f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")
        
        # Write error results
        await write_result_to_files(error_result, output_file, csv_file_name)
        return error_result
        


async def prepare_prompt(prompt: Union[str, List], session_id: Optional[str] = None) -> List[Dict[str, str]]:
    """Prepare prompt with session history if needed and ensure it's in the correct format"""
    # Convert string prompts to proper chat format
    formatted_prompt = []
    
    # If prompt is a string, convert it to a proper chat message
    if isinstance(prompt, str):
        # Check if the prompt starts with a number (as seen in error logs)
        if prompt and prompt[0].isdigit():
            # Remove the first character if it's a digit
            prompt = prompt[1:]
        formatted_prompt = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        # If it's already a list, make sure each item has role and content
        formatted_prompt = prompt
        # Validate each message in the list
        for i, msg in enumerate(formatted_prompt):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                # Convert invalid messages to proper format
                formatted_prompt[i] = {"role": "user", "content": str(msg)}
    else:
        # For any other type, convert to string and make it a user message
        formatted_prompt = [{"role": "user", "content": str(prompt)}]
    
    # Add session history if needed
    if session_id is not None:
        async with history_lock:
            if session_id not in session_history:
                session_history[session_id] = []
            else:
                # Combine history with current messages
                formatted_prompt = session_history[session_id] + formatted_prompt
    
    # Validate final prompt format to ensure it's correct
    if not formatted_prompt:
        # Provide a default message if somehow we ended up with an empty prompt
        formatted_prompt = [{"role": "user", "content": "Hello"}]
    
    logger.debug(f"Formatted prompt: {formatted_prompt}")
    return formatted_prompt

async def update_response(response: str, session_id: Optional[str] = None, history_lock=None):
    """Update session history with response"""
    if session_id is None:
        return
    
    async with history_lock:
        if session_id not in session_history:
            session_history[session_id] = []
        
        # Add user message and assistant response to history
        # Assuming the last prompt was added before this response
        session_history[session_id].append({"role": "assistant", "content": response})

def patch_openai_client(client):
    """Patch the OpenAI client to capture headers from responses"""
    transport = HeaderCaptureTransport()
    # Access the internal httpx client and modify its transport
    if hasattr(client, "_client"):
        client._client._transport = transport
    elif hasattr(client, "client"):
        client.client._transport = transport
    
    return transport

def extract_headers_data(headers):
    """Extract and parse all relevant headers from the response"""
    # Basic headers
    selected_pod_ip = headers.get('target-pod', 'Not Found')
    selected_pod_name = headers.get('target-pod-name', 'Not Found')
    
    # Log missing important headers
    if not selected_pod_name:
        logger.warning("target-pod-name header not found in response")
    if not selected_pod_ip:
        logger.warning("target-pod header not found in response")
    
    # Timing headers with defaults
    gateway_side_ttft = float(headers.get('x-timing-ttft-ms', -1))
    gateway_side_tpot = float(headers.get('x-timing-tpot-ms', -1))
    gateway_side_e2e_latency = float(headers.get('x-timing-e2e-ms', -1))
    kv_cache_hit_ratio = float(headers.get('x-kvcache-hit-ratio', -1))
    
    # Parse JSON headers safely
    def parse_json_header(header_name):
        if header_name in headers:
            try:
                return json.loads(headers.get(header_name))
            except json.JSONDecodeError:
                logger.warning(f"Could not parse {header_name} header")
        return "Not Found"
    
    # Complex JSON headers
    all_pods_kv_cache_hit_ratio = parse_json_header('x-kvcache-hit-ratio-all')
    all_pods_num_inflight_requests = parse_json_header('x-num-inflight-requests-all')
    vllm_gpu_kv_cache_usage = parse_json_header('x-vllm-gpu-kvcache-usage')
    vllm_cpu_kv_cache_usage = parse_json_header('x-vllm-cpu-kvcache-usage')
    vllm_num_running_requests = parse_json_header('x-vllm-num-running-requests')
    vllm_num_waiting_requests = parse_json_header('x-vllm-num-waiting-requests')
    
    return {
        "selected_pod_ip": selected_pod_ip,
        "selected_pod_name": selected_pod_name,
        "gateway_side_ttft": gateway_side_ttft,
        "gateway_side_tpot": gateway_side_tpot,
        "gateway_side_e2e_latency": gateway_side_e2e_latency,
        "kv_cache_hit_ratio": kv_cache_hit_ratio,
        "kv_cache_hit_ratio_all": all_pods_kv_cache_hit_ratio,
        "num_inflight_requests": all_pods_num_inflight_requests,
        "vllm_gpu_kv_cache_usage": vllm_gpu_kv_cache_usage,
        "vllm_cpu_kv_cache_usage": vllm_cpu_kv_cache_usage,
        "vllm_num_running_requests": vllm_num_running_requests,
        "vllm_num_waiting_requests": vllm_num_waiting_requests
    }

def calculate_slo_metrics(prompt_tokens, output_tokens, gateway_side_ttft, gateway_side_tpot, gateway_side_e2e_latency):
    """Calculate SLO metrics based on token counts and latencies"""
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

def create_success_result(request_id, start_time, response_time, client_side_ttft, client_side_tpot, 
                          prompt_tokens, output_tokens, total_tokens, headers_data, 
                          prompt_text, output_text, session_id=None):
    """Create a result dictionary for successful requests"""
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
        "kv_cache_hit_ratio": headers_data["kv_cache_hit_ratio_all"],
        # "num_inflight_requests": headers_data["num_inflight_requests"],
        # "vllm_gpu_kv_cache_usage": headers_data["vllm_gpu_kv_cache_usage"],
        # "vllm_cpu_kv_cache_usage": headers_data["vllm_cpu_kv_cache_usage"],
        # "vllm_num_running_requests": headers_data["vllm_num_running_requests"],
        # "vllm_num_waiting_requests": headers_data["vllm_num_waiting_requests"],
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
        # "prompt_text": prompt_text,
        # "output_text": output_text,
        "error_type": None,
        "error_message": None,
        # "error_traceback": None,
        "session_id": session_id,
    }
    return result

def create_error_result(request_id, start_time, error_time, e, prompt, selected_pod_ip="", selected_pod_name="", session_id=None):
    """Create a result dictionary for failed requests"""
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
        # "prompt_text": prompt,
        # "output_text": None,
        "error_type": error_type,
        "error_message": str(e),
        # "error_traceback": traceback.format_exc(),
        "session_id": session_id,
    }
    
    logger.error(f"Request {request_id}: Error ({error_type}): {str(e)}")
    logger.error(traceback.format_exc())
    
    return result

async def write_result_to_files(result_data, output_file, csv_file, results_lock):
    """Write results to output and CSV files with async locking"""
    if csv_file is None and csv_file_name == "":
        raise ValueError("CSV file path not specified")
    
    # Use async lock to ensure thread safety
    async with results_lock:
        # Write to output file (JSON lines)
        if output_file:
            output_line = json.dumps(result_data) + "\n"
            if isinstance(output_file, io.StringIO):
                output_file.write(output_line)
            else:
                output_file.write(output_line)
                await asyncio.to_thread(output_file.flush)  # Flush using a thread to avoid blocking
        
        # Write to CSV file
        csv_path = csv_file if csv_file else csv_file_name
        if csv_path:
            try:
                # Check if file exists and has content
                file_exists = os.path.exists(csv_path)
                is_new_file = not file_exists or os.path.getsize(csv_path) == 0
                
                # Prepare row data
                csv_row = {}
                for key, value in result_data.items():
                    if isinstance(value, (dict, list)):
                        csv_row[key] = json.dumps(value)
                    else:
                        csv_row[key] = value
                
                # Use a thread for file I/O operations to avoid blocking the event loop
                await asyncio.to_thread(write_csv_row, 
                                      csv_path, 
                                      csv_row, 
                                      is_new_file)
            except Exception as e:
                logger.error(f"Error writing to CSV: {e}")
                logger.error(traceback.format_exc())

def write_csv_row(csv_path, row_data, is_new_file):
    """Helper function to write CSV rows in a separate thread"""
    mode = 'w' if is_new_file else 'a'
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        fieldnames = list(row_data.keys())  # Get the keys in the current order
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row_data)

async def create_client(api_key, endpoint, max_retries, timeout, routing_strategy):
    """Create an OpenAI client instance"""
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

async def send_request_batch(client, model, prompt, output_file, request_id, 
                             session_id=None, target_time=None, max_tokens=2048,
                             temperature=0.0, routing_strategy=None):
    """Send a batch (non-streaming) request asynchronously"""
    start_time = asyncio.get_running_loop().time()
    selected_pod_ip = ""
    selected_pod_name = ""
    client_side_ttft = -1
    client_side_tpot = -1
    scheduled_time = target_time
    actual_start_time = time.time()
    
    try:
        # If target_time is provided, wait until that time
        if target_time is not None:
            current_time = time.time()
            if current_time < target_time:
                schedule_delay = target_time - current_time
                logger.info(f"Request {request_id}: Scheduled for {time.strftime('%H:%M:%S.%f', time.localtime(target_time))[:-3]}, " 
                          f"waiting {schedule_delay:.3f}s")
                await asyncio.sleep(schedule_delay)
            
            # Record the actual start time after waiting
            actual_start_time = time.time()
            scheduling_accuracy = actual_start_time - target_time
            logger.info(f"Request {request_id}: Starting batch request at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]}, "
                      f"scheduling accuracy: {scheduling_accuracy:.6f}s")
        else:
            logger.info(f"Request {request_id}: Starting batch request at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]} (no scheduled time)")
        
        # Double-check prompt format
        if not isinstance(prompt, list):
            # Convert to list format for chat completions
            prompt = [{"role": "user", "content": str(prompt)}]
        elif not prompt:
            prompt = [{"role": "user", "content": "Hello"}]
        
        # Ensure each item in the list has role and content
        for i, msg in enumerate(prompt):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                prompt[i] = {"role": "user", "content": str(msg)}
        
        # Format validation logging
        logger.debug(f"Request {request_id}: Formatted prompt: {prompt}")
        
        # Set additional headers if needed
        extra_headers = {}
        if routing_strategy:
            extra_headers["routing-strategy"] = routing_strategy

        # Patch the client to capture headers
        transport = patch_openai_client(client)

        try:
            # Send request using the OpenAI client
            response = await client.chat.completions.create(
                model=model,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=extra_headers
            )
            
            # Validate response
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Incomplete or invalid response received")

            # Extract headers data
            headers_data = extract_headers_data(transport.captured_headers)
            print(f"Request {request_id}, headers_data: {headers_data}")
            # Extract response time and token counts
            response_time = asyncio.get_running_loop().time()
            completion_time = time.time()
            prompt_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            output_text = response.choices[0].message.content
            
            # Update session history if needed
            if session_id:
                await update_response(output_text, session_id)

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
                prompt_text=prompt,
                output_text=output_text,
                session_id=session_id
            )
            
            # Calculate total elapsed time
            total_elapsed = completion_time - actual_start_time
            
            logger.info(f"Request {request_id}: Completed at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]}. "
                       f"Elapsed: {total_elapsed:.3f}s, "
                       f"Tokens: {prompt_tokens} in / {output_tokens} out, "
                       f"E2E latency: {float(result['client_side_e2e_latency_in_ms']):.2f}ms")
            
            # Log scheduling information
            if scheduled_time:
                scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
                actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
                logger.info(f"Request {request_id}: Scheduling summary - "
                          f"Scheduled: {scheduled_dt}, "
                          f"Started: {actual_dt}, "
                          f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")
            
            # Write results to files
            await write_result_to_files(result, output_file, csv_file_name)
            return result
            
        except openai.BadRequestError as e:
            # Specific handling for format errors
            logger.error(f"Request {request_id}: Bad request error: {str(e)}")
            # Attempt to get error details
            error_msg = str(e)
            
            # If the error is related to message format, retry with a simplified format
            if "messages" in error_msg and "Input should be a valid list" in error_msg:
                logger.warning(f"Request {request_id}: Message format error detected, retrying with simplified format")
                # Extract just the text content and retry with a simplified format
                try:
                    simple_content = "".join([msg.get("content", "") for msg in prompt if isinstance(msg, dict)])
                    if not simple_content:
                        simple_content = str(prompt)
                    
                    simple_prompt = [{"role": "user", "content": simple_content}]
                    logger.debug(f"Request {request_id}: Retrying with simplified prompt: {simple_prompt}")
                    
                    # Retry with simplified format
                    response = await client.chat.completions.create(
                        model=model,
                        messages=simple_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        extra_headers=extra_headers
                    )
                    
                    # Process response as before
                    if not response or not hasattr(response, 'choices') or not response.choices:
                        raise ValueError("Incomplete or invalid response received on retry")
                    
                    headers_data = extract_headers_data(transport.captured_headers)
                    response_time = asyncio.get_running_loop().time()
                    completion_time = time.time()
                    prompt_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    output_text = response.choices[0].message.content
                    
                    if session_id:
                        await update_response(output_text, session_id)
                    
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
                        prompt_text=simple_prompt,
                        output_text=output_text,
                        session_id=session_id
                    )
                    
                    # Calculate total elapsed time
                    total_elapsed = completion_time - actual_start_time
                    
                    logger.info(f"Request {request_id}: Completed on retry at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]}. "
                              f"Elapsed: {total_elapsed:.3f}s, "
                              f"Tokens: {prompt_tokens} in / {output_tokens} out, "
                              f"E2E latency: {float(result['client_side_e2e_latency_in_ms']):.2f}ms")
                    
                    # Log scheduling information
                    if scheduled_time:
                        scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
                        actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
                        logger.info(f"Request {request_id}: Scheduling summary - "
                                  f"Scheduled: {scheduled_dt}, "
                                  f"Started: {actual_dt}, "
                                  f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")
                    
                    # Write results to files
                    await write_result_to_files(result, output_file, csv_file_name)
                    return result
                    
                except Exception as retry_e:
                    # If retry failed, continue to error handling
                    logger.error(f"Request {request_id}: Retry failed: {str(retry_e)}")
                    raise retry_e
            
            # If we're here, either it wasn't a format error or the retry failed
            raise e

    except Exception as e:
        error_time = asyncio.get_running_loop().time()
        completion_time = time.time()
        
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
        
        # Calculate total elapsed time
        total_elapsed = completion_time - actual_start_time
        
        logger.error(f"Request {request_id}: Error at {time.strftime('%H:%M:%S.%f', time.localtime(completion_time))[:-3]} "
                   f"after {total_elapsed:.3f}s: {error_result['error_type']}: {error_result['error_message']}")
        
        # Log scheduling information for errors too
        if scheduled_time:
            scheduled_dt = time.strftime('%H:%M:%S.%f', time.localtime(scheduled_time))[:-3]
            actual_dt = time.strftime('%H:%M:%S.%f', time.localtime(actual_start_time))[:-3]
            logger.error(f"Request {request_id}: Scheduling error summary - "
                      f"Scheduled: {scheduled_dt}, "
                      f"Started: {actual_dt}, "
                      f"Variance: {(actual_start_time - scheduled_time)*1000:.2f}ms")
        
        # Write error results
        await write_result_to_files(error_result, output_file, csv_file_name)
        return error_result



async def schedule_and_execute_tasks(tasks, client, model, is_streaming, output_file, max_tokens, temperature, routing_strategy, results_lock, history_lock):
    """Schedule and execute tasks based on their target times with true concurrency"""
    # Sort tasks by target_time
    tasks.sort(key=lambda t: t["target_time"])
    
    # Select the appropriate send function based on streaming mode
    send_func = send_request_streaming if is_streaming else send_request_batch
    
    # Create a list to hold all task futures
    all_task_futures = []
    
    # Current time reference
    base_time = time.time()
    logger.info(f"Base time for scheduling: {time.strftime('%H:%M:%S.%f', time.localtime(base_time))[:-3]}")
    
    # Create a task for each request with its own scheduled execution time
    for task in tasks:
        target_time = task["target_time"]
        delay = max(0, target_time - base_time)
        
        # Create a scheduled task using asyncio
        scheduled_task = asyncio.create_task(
            schedule_task(
                delay=delay,
                target_time=target_time,
                request_id=task["request_id"],
                send_func=send_func,
                client=client,
                model=model,
                prompt=task["prompt"],
                output_file=output_file,
                session_id=task["session_id"],
                max_tokens=max_tokens,
                temperature=temperature,
                routing_strategy=routing_strategy,
                results_lock=results_lock,
                history_lock=history_lock,
            )
        )
        
        all_task_futures.append(scheduled_task)
    
    logger.info(f"Scheduled {len(all_task_futures)} tasks for concurrent execution")
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*all_task_futures, return_exceptions=True)
    
    # Process results
    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    failure_count = len(tasks) - success_count
    
    logger.info(f"All tasks completed: {success_count} successful, {failure_count} failed")
    
    return results

async def schedule_task(delay, target_time, request_id, send_func, client, model, prompt, 
                        output_file, session_id, max_tokens, temperature, routing_strategy, results_lock, history_lock):
    """Schedule and execute a single task at the specified time"""
    task_start = time.time()
    
    # Wait until the scheduled time
    if delay > 0:
        logger.debug(f"Request {request_id}: Waiting {delay:.3f}s until scheduled time {time.strftime('%H:%M:%S.%f', time.localtime(target_time))[:-3]}")
        await asyncio.sleep(delay)
    
    # Record actual start time after waiting
    actual_start = time.time()
    wait_accuracy = actual_start - (task_start + delay)
    
    logger.debug(f"Request {request_id}: Executing at {time.strftime('%H:%M:%S.%f', time.localtime(actual_start))[:-3]}, " 
               f"scheduling accuracy: {(actual_start - target_time)*1000:.2f}ms, wait accuracy: {wait_accuracy*1000:.2f}ms")
    
    # Execute the task
    result = await send_func(
        client=client,
        model=model,
        prompt=prompt,
        output_file=output_file,
        request_id=request_id,
        session_id=session_id,
        target_time=target_time,  # No additional waiting as we've already done that
        max_tokens=max_tokens,
        temperature=temperature,
        routing_strategy=routing_strategy,
        results_lock=results_lock,
        history_lock=history_lock,
    )
    
    return result

async def run_benchmark(api_key, endpoint, max_retries, timeout, routing_strategy,
                       load_struct, output_file, model, max_tokens,
                       temperature, is_streaming, results_lock, history_lock):
    """Main benchmark function that runs all requests asynchronously"""
    # Create a client
    client = await create_client(api_key, endpoint, max_retries, timeout, routing_strategy)
    
    # Base time for scheduling
    base_time = time.time()
    
    # Prepare all tasks
    all_tasks = []
    request_id = 0
    
    # Process the load structure and create tasks
    for requests_dict in load_struct:
        ts = int(requests_dict["timestamp"])
        requests = requests_dict["requests"]
        target_time = base_time + ts / 1000.0  # Convert milliseconds to seconds
        
        for request in requests:
            session_id = request.get("session_id", None)
            prompt = await prepare_prompt(prompt=request["prompt"], session_id=session_id)
            
            task = {
                "prompt": prompt,
                "request_id": request_id,
                "session_id": session_id,
                "target_time": target_time
            }
            all_tasks.append(task)
            request_id += 1
    
    logger.info(f"Scheduling {len(all_tasks)} tasks for execution")
    
    # Execute all tasks with true concurrency
    start_time = time.time()
    results = await schedule_and_execute_tasks(
        tasks=all_tasks,
        client=client,
        model=model,
        is_streaming=is_streaming,
        output_file=output_file,
        max_tokens=max_tokens,
        temperature=temperature,
        routing_strategy=routing_strategy,
        results_lock=results_lock,
        history_lock=history_lock,
    )
    end_time = time.time()
    
    # Log benchmark completion
    logger.info(f"Benchmark completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Total requests: {len(all_tasks)}")
    
    # Count successes and failures
    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    error_count = len(all_tasks) - success_count
    
    logger.info(f"Successful requests: {success_count}")
    logger.info(f"Failed requests: {error_count}")
    
    return results



async def main(args):
    global csv_file_name
    if '.jsonl' not in args.workload_path:
        raise ValueError("Workload path must be a .jsonl file")
    csv_file_name = f"{args.workload_path.replace('.jsonl', '')}"
    csv_file_name += f"-maxtoken{args.max_tokens}"
    csv_file_name += ".output.csv"
    
    # Initialize CSV file
    with open(csv_file_name, 'w', encoding='utf-8') as f:
        f.write("")  # Create empty file
    
    # Load workload
    load_struct = await load_workload(args.workload_path)
    
    results_lock = asyncio.Lock()  # Async lock for result writing
    history_lock = asyncio.Lock()  # Async lock for session history

    # Open output file
    with open(args.output_file_path, 'w', encoding='utf-8') as output_file:
        # Run benchmark
        start_time = time.time()
        await run_benchmark(
            api_key=args.api_key,
            endpoint=args.endpoint,
            max_retries=args.max_retries,
            timeout=args.timeout,
            routing_strategy=args.routing_strategy,
            load_struct=load_struct,
            output_file=output_file,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            is_streaming=args.streaming,
            results_lock=results_lock,
            history_lock=history_lock,
        )
        end_time = time.time()
        
        logger.info(f"Total benchmark time: {end_time - start_time:.2f} seconds")
        print(f"** csv_file_name: {csv_file_name}")

def write_experiment_config_to_file(output_dir, args):
        config_file = f'{output_dir}/experiment_config.txt'
        with open(config_file, 'w') as f:
            f.write("Experiment Configuration:\n")
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")
        return config_file

if __name__ == "__main__":
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Async Workload Generator')
    parser.add_argument("--workload_path", type=str, required=True, help="File path to the workload file.")
    parser.add_argument("--model", type=str, required=True, help="Default target model.")
    parser.add_argument('--endpoint', type=str, required=True, help="API endpoint URL.")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the service.")
    parser.add_argument('--output_file_path', type=str, default="output.jsonl", help="Output file path for JSON results.")
    parser.add_argument("--streaming", action="store_true", help="Use streaming client.")
    parser.add_argument("--routing_strategy", type=str, default="random", help="Routing strategy to use.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for the request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the request.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=0, help="Maximum number of retries for failed requests.")
    args = parser.parse_args()

    utils.restart_deploy('aibrix-gateway-plugins', 'aibrix-system')
    # utils.restart_deploy('llama-3-8b-instruct', 'default')
    time.sleep(3)
    utils.check_deployment_ready_kubernetes('aibrix-gateway-plugins', 'aibrix-system')
    utils.check_deployment_ready_kubernetes('llama-3-8b-instruct', 'default')

    asyncio.run(main(args))
    
    workload_name = args.workload_path.split("/")[-1].split(".")[0]
    output_dir = f'output/{workload_name}-maxtokens{args.max_tokens}-{utils.get_current_pdt_time()}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config_file = write_experiment_config_to_file(output_dir, args)
    gatway_log_file_name = f'{output_dir}/gateway-plugins.log.csv'
    success = utils.collect_k8s_logs(
        namespace='aibrix-system',
        deployment_name='aibrix-gateway-plugins',
        output_file=gatway_log_file_name,
        keyword='**@latency_metrics'
    )