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

thread_pool_size = 8
QUEUE_SIZE = thread_pool_size * 2
logging.basicConfig(level=logging.INFO)
task_queue = Queue(maxsize=QUEUE_SIZE)
session_history = {}
lock = threading.Lock()

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



async def send_request_streaming(client: openai.AsyncOpenAI,
                             model: str,
                             prompt: str,
                             output_file: str,
                             request_id: int,
                             session_id: str,
                             target_time: int,
                             ):
    start_time = asyncio.get_event_loop().time()
    first_response_time = None
    target_pod = ""
    target_request_id = ""
    try:
        cur_time = time.time()
        logging.warning(f"send_request_streaming: Prepare to launch task after {target_time - cur_time}")
        response_stream = await client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0,
            max_tokens=2048,
            stream=True,
            stream_options={"include_usage": True},
        )
        if hasattr(response_stream, 'response') and hasattr(response_stream.response, 'headers'):
            target_pod = response_stream.response.headers.get('target-pod')
            target_request_id = response_stream.response.headers.get('request-id')

        text_chunks = []
        prompt_tokens = 0
        output_tokens = 0
        total_tokens = 0

        try:
            async for chunk in response_stream:
                if chunk.choices:
                    if chunk.choices[0].delta.content is not None:
                        if not first_response_time:
                            first_response_time = asyncio.get_event_loop().time()
                        output_text = chunk.choices[0].delta.content
                        text_chunks.append(output_text)
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    # For OpenAI, we expect to get complete usage stats, not partial ones to accumulate
                    # So we can safely overwrite previous values if they exist
                    if chunk.usage.prompt_tokens is not None:
                        prompt_tokens = chunk.usage.prompt_tokens
                    if chunk.usage.completion_tokens is not None:
                        output_tokens = chunk.usage.completion_tokens
                    if chunk.usage.total_tokens is not None:
                        total_tokens = chunk.usage.total_tokens
        except Exception as stream_error:
            # Handle errors during streaming
            logging.error(f"Request {request_id}: Stream interrupted: {type(stream_error).__name__}: {str(stream_error)}")

        response_text = "".join(text_chunks)
        response_time = asyncio.get_event_loop().time()
        latency = response_time - start_time
        throughput = output_tokens / latency if output_tokens > 0 else 0
        ttft = first_response_time - start_time if first_response_time else None
        tpot = (response_time - first_response_time) / output_tokens if first_response_time and output_tokens > 0 else None

        update_response(response = response_text, lock = lock, session_id = session_id, history = session_history)
        
        result = {
            "request_id": request_id,
            "status": "success",
            "input": prompt,
            "output": response_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
            "throughput": throughput,
            "start_time": start_time,
            "end_time": response_time,
            "ttft": ttft,
            "tpot": tpot,
            "target_pod": target_pod,
            "target_request_id": target_request_id,
            "session_id": session_id,
        }

        # Write result to JSONL file
        logging.info(f"Request {request_id}: Completed successfully. Tokens: {total_tokens}, Latency: {latency:.2f}s")
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()  # Ensure data is written immediately to the file
        return result

    except Exception as e:
        error_time = asyncio.get_event_loop().time()
        # Determine error type based on exception class
        error_type = type(e).__name__
        error_result = {
            "request_id": request_id,
            "status": "error",
            "error_type": error_type,
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "input": prompt,
            "latency": error_time - start_time,
            "start_time": start_time,
            "end_time": error_time,
            "target_pod": target_pod,
            "target_request_id": target_request_id,
            "session_id": session_id,
        }
        logging.error(f"Request {request_id}: Error ({error_type}): {str(e)}")
        output_file.write(json.dumps(error_result) + "\n")
        output_file.flush()
        return error_result

async def benchmark_streaming(api_key: str,
                              endpoint: str,
                              max_retries: int,
                              timeout: float,
                              routing_strategy: str,
                              load_struct: List,
                              output_file: io.TextIOWrapper,
                              model: str):
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
            task_queue.put((formatted_prompts[i], output_file, request_id, session_id, target_time))
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
    selected_pod = ""
    csv_file_name = 'output.csv'
    if request_id == 0:
        csv_file = open(csv_file_name, 'w', encoding='utf-8')
    try:
        if target_time is not None:
            cur_time = time.time()
            if target_time > cur_time:
                await asyncio.sleep(target_time - cur_time)
                
        logging.info(f"Request {request_id}: Starting sending request to {endpoint}")
        
        # Extract the content from the message format
        prompt_content = prompt[0]["content"] if isinstance(prompt, list) and len(prompt) > 0 and "content" in prompt[0] else prompt
        
        # Use aiohttp for direct HTTP request to match curl format
        headers = {
            "Content-Type": "application/json",
            "model": model,
            "routing-strategy": routing_strategy
        }
        
        # Add Authorization if API key is provided
        if hasattr(client, "api_key") and client.api_key:
            headers["Authorization"] = f"Bearer {client.api_key}"
            
        # Construct payload in the exact format expected by the server
        max_tokens = 2048 if max_tokens is None else max_tokens
        payload = {
            "model": model,  # Include model in the body too
            "prompt": prompt_content,
            "temperature": 0,
            "max_tokens": max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{endpoint}/v1/chat/completions", 
                                headers=headers, 
                                json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Request failed with status {response.status}: {await response.text()}")
                
                headers = response.headers
                all_headers = dict(headers)  # Convert to regular dictionary
                logging.info(f"Request {request_id}: Response headers:")
                for header_name, header_value in all_headers.items():
                    logging.info(f"  {header_name}: {header_value}")
                
                response_data = await response.json()
                selected_pod_ip = response.headers.get('target-pod', '')
                selected_pod_name = response.headers.get('target-pod-name', '')
                if selected_pod_name == "":
                    logging.error(f"target-pod-name header not found in response.")
                if selected_pod_ip == "":
                    logging.error(f"target-pod header not found in response.")
                
                response_time = asyncio.get_event_loop().time()
                client_side_e2e_latency_in_ms = (response_time - start_time) * 1000
                
                # Extract tokens from response - adjust this based on your API's response format
                prompt_tokens = int(response_data.get("usage", {}).get("prompt_tokens", None))
                output_tokens = int(response_data.get("usage", {}).get("completion_tokens", None))
                total_tokens = int(response_data.get("usage", {}).get("total_tokens", None))

                gateway_side_ttft = float(response.headers.get('x-timing-ttft-ms', None))
                gateway_side_tpot = float(response.headers.get('x-timing-tpot-ms', None))
                gateway_side_e2e_latency = float(response.headers.get('x-timing-e2e-ms', None)) 
                kv_cache_hit_ratio = float(response.headers.get('x-kvcache-hit-ratio', None))
                all_pods_kv_cache = {}
                if 'x-kvcache-hit-ratio-all' in headers:
                    try:
                        all_pods_kv_cache = json.loads(headers.get('x-kvcache-hit-ratio-all'))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse x-kvcache-hit-ratio-all header")
                else:
                    all_pods_kv_cache = None
                if 'x-num-inflight-requests-all' in headers:
                    try:
                        all_pods_inflight = json.loads(headers.get('x-num-inflight-requests-all'))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse x-num-inflight-requests-all header")
                else:
                    all_pods_inflight = None

                if 'x-vllm-gpu-kvcache-usage' in headers:
                    try:
                        vllm_gpu_kv_cache_usage = json.loads(headers.get('x-vllm-gpu-kvcache-usage'))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse x-vllm-gpu-kvcache-usage header")
                else:
                    vllm_gpu_kv_cache_usage = None
                if 'x-vllm-cpu-kvcache-usage' in headers:
                    try:
                        vllm_cpu_kv_cache_usage = json.loads(headers.get('x-vllm-cpu-kvcache-usage'))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse x-vllm-cpu-kvcache-usage header")
                else:
                    vllm_cpu_kv_cache_usage = None
                if 'x-vllm-num-running-requests' in headers:
                    try:
                        vllm_num_running_requests = json.loads(headers.get('x-vllm-num-running-requests'))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse x-vllm-num-running-requests header")
                else:
                    vllm_num_running_requests = None
                if 'x-vllm-num-waiting-requests' in headers:
                    try:
                        vllm_num_waiting_requests = json.loads(headers.get('x-vllm-num-waiting-requests'))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse x-vllm-num-waiting-requests header")
                else:
                    vllm_num_waiting_requests = None

                per_token_ttft_slo_in_ms = 1
                per_token_tpot_slo_in_ms = 10
                ttft_slo_in_ms = per_token_ttft_slo_in_ms * prompt_tokens
                tpot_slo_in_ms = per_token_tpot_slo_in_ms * output_tokens
                e2e_slo_in_ms = ttft_slo_in_ms + tpot_slo_in_ms
                e2e_slo_satisfied = gateway_side_e2e_latency <= e2e_slo_in_ms
                ttft_slo_satisfied = gateway_side_ttft <= ttft_slo_in_ms
                tpot_slo_satisfied = gateway_side_tpot <= tpot_slo_in_ms
                throughput = output_tokens / client_side_e2e_latency_in_ms if output_tokens > 0 else 0
                
                # Extract output text based on response format
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    if "message" in response_data["choices"][0]:
                        output_text = response_data["choices"][0]["message"].get("content", "")
                    elif "text" in response_data["choices"][0]:
                        output_text = response_data["choices"][0]["text"]
                    else:
                        output_text = str(response_data["choices"][0])
                else:
                    output_text = str(response_data)

        result = {
            "request_id": request_id,
            "status": "success",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "client_side_token_per_second": f"{throughput:.2f}",
            "client_side_start_time": f"{start_time:.2f}",
            "client_side_end_time": f"{response_time:.2f}",
            "client_side_e2e_latency_in_ms": f"{client_side_e2e_latency_in_ms:.4f}",
            "client_side_ttft": "Unknown",
            "client_side_tpot": "Unknown",
            "gateway_side_ttft": gateway_side_ttft,
            "gateway_side_tpot": gateway_side_tpot,
            "gateway_side_e2e_latency": gateway_side_e2e_latency,
            "selected_pod_ip": selected_pod_ip,
            "selected_pod_name": selected_pod_name,
            "gpu_model": "NVIDIA-L20",
            "kv_cache_hit_ratio": all_pods_kv_cache,
            "num_inflight_requests": all_pods_inflight,
            'vllm_gpu_kv_cache_usage': vllm_gpu_kv_cache_usage,
            'vllm_cpu_kv_cache_usage': vllm_cpu_kv_cache_usage,
            'vllm_num_running_requests': vllm_num_running_requests,
            'vllm_num_waiting_requests': vllm_num_waiting_requests,
            "e2e_slo_in_ms": e2e_slo_in_ms,
            "ttft_slo_in_ms": ttft_slo_in_ms,
            "tpot_slo_in_ms": tpot_slo_in_ms,
            "e2e_slo_satisfied": e2e_slo_satisfied,
            "ttft_slo_satisfied": ttft_slo_satisfied,
            "tpot_slo_satisfied": tpot_slo_satisfied,
            "error_type": None,
            "error_message": None,
            "error_traceback": None,
            "input": prompt
        }
        
        logging.info(f"Request {request_id}: Completed successfully. Input tokens: {prompt_tokens}, Output tokens: {output_tokens}, Total tokens: {total_tokens}, client_side_e2e_latency_in_ms: {client_side_e2e_latency_in_ms:.2f}s")
        
        # Write results to files
        write_result_to_files(result, output_file, csv_file_name)
        
        return result

    except Exception as e:
        error_time = asyncio.get_event_loop().time()
        error_type = type(e).__name__
        
        # Create error result with the same structure as success result
        # Set all fields that would be in a success result to None for consistency
        error_result = {
            "request_id": request_id,
            "status": "error",
            "prompt_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "client_side_token_per_second": None,
            "client_side_start_time": f"{start_time:.2f}",
            "client_side_end_time": f"{error_time:.2f}",
            "client_side_e2e_latency_in_ms": f"{error_time - start_time:.4f}",
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
            # Add error-specific fields
            "error_type": error_type,
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "input": prompt
        }
        
        logging.error(f"Request {request_id}: Error ({error_type}): {str(e)}")
        logging.error(f"traceback.format_exc(): {traceback.format_exc()}")
        
        # Write error results to files using the same helper function
        write_result_to_files(error_result, output_file, csv_file_name)
        
        return error_result

# Asynchronous request handler
async def send_request_batch(client: openai.AsyncOpenAI,
                             model: str,
                             endpoint: str,
                             prompt: str,
                             output_file: str,
                             request_id: int,
                             session_id: str, 
                             target_time: int,
                             ):
    start_time = asyncio.get_event_loop().time()
    target_pod = ""
    try:
        cur_time = time.time()
        logging.warning(f"send_request_batch: Prepare to launch task after {target_time - cur_time}")
        if target_time > cur_time:
            await asyncio.sleep(target_time - cur_time)
        response = await client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0,
            max_tokens=2048
        )
        if hasattr(response, 'response') and hasattr(response.response, 'headers'):
            target_pod = response.response.headers.get('target-pod')

        response_time = asyncio.get_event_loop().time()
        latency = response_time - start_time
        prompt_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        throughput = output_tokens / latency
        output_text = response.choices[0].message.content

        update_response(response = output_text, lock = lock, session_id = session_id, history = session_history)
        
        result = {
            "request_id": request_id,
            "status": "success",
            # "input": prompt,
            # "output": output_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency": f"{latency:.4f}",
            "client_side_token_per_second": f"{throughput:.2f}",
            "start_time": f"{start_time:.2f}",
            "end_time": f"{response_time:.2f}",
            "ttft": "Unknown",
            "tpot": "Unknown",
            "target_pod": target_pod,
            "session_id": session_id,
        }
        logging.info(f"Request {request_id}: Completed successfully. Tokens: {total_tokens}, Latency: {latency:.2f}s")
        # Write result to JSONL file
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()  # Ensure data is written immediately to the file
        return result

    except Exception as e:
        error_time = asyncio.get_event_loop().time()
        error_type = type(e).__name__
        error_result = {
            "request_id": request_id,
            "status": "error",
            "error_type": error_type,
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "input": prompt,
            "latency": error_time - start_time,
            "start_time": start_time,
            "end_time": error_time,
            "target_pod": target_pod,
            "session_id": session_id,
        }
        logging.error(f"Request {request_id}: Error ({error_type}): {str(e)}")
        output_file.write(json.dumps(error_result) + "\n")
        output_file.flush()
        return error_result


async def benchmark_batch(api_key: str,
                          endpoint: str,
                          max_retries: int,
                          timeout: float,
                          routing_strategy: str,
                          load_struct: List,
                          output_file: io.TextIOWrapper,
                          model: str,
                          max_tokens: int = None):
    request_id = 0
    base_time = time.time()
    num_requests = 0
    threads = []
    
    for thread_idx in range(0, thread_pool_size):
        client = create_client(api_key, endpoint, max_retries, timeout, routing_strategy)
        threads.append(start_worker_threads(thread_idx, client, model, send_request_batch_for_mock_app_format, output_file))
    
    for requests_dict in load_struct:
        ts = int(requests_dict["timestamp"])
        requests = requests_dict["requests"]
        target_time = base_time + ts / 1000.0
        formatted_prompts = [prepare_prompt(prompt = request["prompt"], lock = lock, session_id = request.get("session_id", None), history = session_history) for request in requests]
        
        for i in range(len(requests)):
            session_id = requests[i].get("session_id", None)
            task_queue.put((endpoint, formatted_prompts[i], output_file, request_id, max_tokens, routing_strategy, target_time))
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
                max_tokens = args.max_tokens
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

    args = parser.parse_args()
    main(args)
