import argparse
from tabulate import tabulate
from transformers import AutoConfig, AutoModel
import csv
import os
import json
import requests
from bs4 import BeautifulSoup
import math


"""
Assume that the total compute is shared evenly between requests
Given the model name
Given the number of concurrent request
// Given the context length of model user wants to serve

Case 1:
Given the prefill time target per request

Case 2:
Given the Time-per-Output-Token (TPOT) target per request

Case 3:
Given Time-To-First-Token (for all requests) target (need prompt_size)

Case 4:
Given the complete all requests response time (prompt_size + response_size OR model context windows)

Search for the minimum number of GPUs each GPU need to serve the targeted model

"""

def perform_calculations_find_target_performance_criteria(model_list, gpu_specs, target_performance, prompt_size, response_size, n_concurrent_request, avg_context_window, target_case, datatype, check_fit_gpu=False):

    # use_case = "General"
    print(f" target_performance = {target_performance}, prompt_size = {prompt_size} tokens, response_size = {response_size} tokens")
    print(f" n_concurrent_request = {n_concurrent_request}, target_case = {target_case}")
    print(f"datatype: {datatype}- ", datatype == "FP8")
    if datatype == "FP8":
        byte_factor = 1
    else:
        byte_factor = 2


    model_specs = []
    for model in model_list:
        model = model.lstrip().rstrip()
        model_config = AutoConfig.from_pretrained(model)

        url = 'https://huggingface.co/' + model

        selector = 'body > div > main > div.container.relative.flex.flex-col.md\:grid.md\:space-y-0.w-full.md\:grid-cols-12.md\:flex-1.md\:grid-rows-full.space-y-4.md\:gap-6 > section.pt-8.border-gray-100.md\:col-span-5.pt-6.md\:pb-24.md\:pl-6.md\:border-l.order-first.md\:order-none > div:nth-child(3) > div > div.flex.flex-wrap.gap-x-1\\.5.gap-y-1.text-sm > div:nth-child(1) > div.px-1\\.5'

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            element = soup.select_one(selector)
            extracted_text = element.text.strip() if element else ''
            # return jsonify({'text': extracted_text})
            print(extracted_text)
        except Exception as e:
            print("Fail to scrape from webpage")
            # return jsonify({'error': str(e)}), 500


        # Try these in order:
        context_length = getattr(model_config, "max_position_embeddings", None)
        if context_length is None:
            context_length = getattr(model_config, "n_positions", None)
        if context_length is None:
            context_length = getattr(model_config, "max_sequence_length", None)
        if context_length is None:
            context_length = getattr(model_config, "max_seq_len", None)

        if hasattr(model_config, "sliding_window"):
            if model_config.sliding_window is not None:
                context_length = model_config.sliding_window * model_config.num_key_value_heads

        model_specs.append({
            "name": model,
            "params_billion": float(extracted_text.split('B')[0]),
            "d_model": model_config.hidden_size,
            "n_heads": model_config.num_attention_heads,
            "n_layers": model_config.num_hidden_layers,
            "max_context_window": context_length,
            "d_head": model_config.hidden_size // model_config.num_attention_heads,
        })



    BYTES_IN_GB = 1_073_741_824  # 1 GB = 1,073,741,824 bytes

    def calc_kv_cache_size_per_token(n_layers, d_model, byte_factor=2):
        return byte_factor * 2 * n_layers * d_model / BYTES_IN_GB  # GB/token

    def calc_memory_footprint(model_spec, n_concurrent_request, avg_context_window, byte_factor=2):
        kv_cache_size_per_token = calc_kv_cache_size_per_token(model_spec["n_layers"], model_spec["d_model"], byte_factor)
        target_gpu_mem = kv_cache_size_per_token * avg_context_window * n_concurrent_request + model_spec["params_billion"] * byte_factor
        return target_gpu_mem

    # case 1:
    
    def calc_num_gpus_to_meet_prefill_time_per_token(prefill_time_per_token, n_concurrent_request, model_params_billion, fp16_tflops, byte_factor=2):
        num_gpu = (byte_factor * model_params_billion / prefill_time_per_token) / fp16_tflops * n_concurrent_request
        return math.ceil(num_gpu)
    
    # case 2:

    def calc_num_gpus_to_meet_generation_time_per_token(time_per_output_token, n_concurrent_request, model_params_billion, memory_bandwidth_gbps, byte_factor=2):
        num_gpu = (byte_factor * model_params_billion / time_per_output_token) / memory_bandwidth_gbps * 1000 * n_concurrent_request
        return math.ceil(num_gpu)

    # case 3:

    def calc_num_gpus_to_meet_estimated_ttft_time(
        ttft_s, n_concurrent_request, model_params_billion,  prompt_size, fp16_tflops, gpu_memory_gb, memory_bandwidth_gbps, byte_factor=2
    ):
    
        # ttft_s * 1000 = (prompt_size * prefill_time + generation_time)

        # ttft_s * 1000 = prompt_size * (byte_factor * model_params_billion / num_gpu) / fp16_tflops + (byte_factor * model_params_billion / num_gpu) / memory_bandwidth_gbps * 1000

        num_gpus = (prompt_size * (byte_factor * model_params_billion) / fp16_tflops * n_concurrent_request + (byte_factor * model_params_billion) / memory_bandwidth_gbps * n_concurrent_request * 1000) / (ttft_s * 1000)

        return math.ceil(num_gpus)


    # case 4:

    def calc_num_gpus_to_meet_estimated_response_time(
        response_time_s, n_concurrent_request, model_params_billion, prompt_size, response_size, fp16_tflops, gpu_memory_gb, memory_bandwidth_gbps, byte_factor=2
    ):
        # response_time_s * 1000 = (prompt_size * (byte_factor * model_params_billion / num_gpu) / fp16_tflops + response_size * (byte_factor * model_params_billion / num_gpu) / memory_bandwidth_gbps * 1000)

        num_gpus = (prompt_size * (byte_factor * model_params_billion) / fp16_tflops * n_concurrent_request + response_size * (byte_factor * model_params_billion) / memory_bandwidth_gbps * n_concurrent_request * 1000) / (response_time_s * 1000 )

        return math.ceil(num_gpus)

    def calc_kv_cache_tokens(num_gpu, gpu_memory_gb, model_params_billion, kv_cache_size, byte_factor=2):
        result = (num_gpu * gpu_memory_gb - byte_factor * model_params_billion) / kv_cache_size
        return result if result >= 0 else "OOM"

    def calc_prefill_time_per_token(num_gpu, model_params_billion, fp16_tflops, n_concurrent_request, byte_factor=2):
        result = (byte_factor * model_params_billion / num_gpu) / fp16_tflops * n_concurrent_request
        return result if result >= 0 else "OOM"

    def calc_generation_time_per_token(num_gpu, model_params_billion, memory_bandwidth_gbps, n_concurrent_request, byte_factor=2):
        result = (byte_factor * model_params_billion / num_gpu) / memory_bandwidth_gbps * 1000 * n_concurrent_request
        return result if result >= 0 else "OOM"

    def calc_estimated_ttft_time(prefill_time, generation_time, prompt_size, response_size):
        if isinstance(prefill_time, str) or isinstance(generation_time, str):  # Check if any are "NA"
            return "OOM"
        return (prompt_size * prefill_time + generation_time) / 1000  # convert ms to seconds

    def calc_estimated_response_time(prefill_time, generation_time, prompt_size, response_size):
        if isinstance(prefill_time, str) or isinstance(generation_time, str):  # Check if any are "NA"
            return "OOM"
        return (prompt_size * prefill_time + response_size * generation_time) / 1000  # convert ms to seconds

    print(f"\n******************** Estimate LLM Capacity and Latency ******************** ")
    capacity_latency_table = []
    for model in model_specs:
        kv_cache_size = calc_kv_cache_size_per_token(model['n_layers'], model['d_model'], byte_factor)
        
        memory_footprint = calc_memory_footprint(
            model, 
            n_concurrent_request, 
            avg_context_window if avg_context_window > 0 else model["max_context_window"], 
            byte_factor
        )

        for gpu in gpu_specs[datatype]:  # Use the correct GPU specs based on datatype

            num_gpus = -1
            if target_case == 'prefill_time':
                num_gpus = calc_num_gpus_to_meet_prefill_time_per_token(
                    target_performance,
                    n_concurrent_request,
                    model['params_billion'],
                    gpu['fp16_tflops'], 
                    byte_factor
                )
            elif target_case == 'tpot':
                num_gpus = calc_num_gpus_to_meet_generation_time_per_token(
                    target_performance,
                    n_concurrent_request,
                    model['params_billion'], 
                    gpu['memory_bandwidth_gbps'], 
                    byte_factor
                )
            elif target_case == 'ttft':
                num_gpus = calc_num_gpus_to_meet_estimated_ttft_time(
                    target_performance,
                    n_concurrent_request,
                    model['params_billion'], 
                    prompt_size,
                    gpu['fp16_tflops'],
                    gpu['memory_gb'],
                    gpu['memory_bandwidth_gbps'], 
                    byte_factor
                )
            elif target_case == 'response_time':
                num_gpus = calc_num_gpus_to_meet_estimated_response_time(
                    target_performance,
                    n_concurrent_request,
                    model['params_billion'], 
                    prompt_size,
                    response_size,
                    gpu['fp16_tflops'],
                    gpu['memory_gb'],
                    gpu['memory_bandwidth_gbps'], 
                    byte_factor
                )
            else:
                num_gpus = 1
                capacity_latency_table.append(["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
                return capacity_latency_table

            if check_fit_gpu:
                minimum_num_gpu_required = math.ceil(memory_footprint / gpu['memory_gb'])
                num_gpus = max(num_gpus, minimum_num_gpu_required)

            kv_cache_tokens = calc_kv_cache_tokens(num_gpus, gpu['memory_gb'], model['params_billion'], kv_cache_size, byte_factor)
            prefill_time_per_token_per_request = calc_prefill_time_per_token(num_gpus, model['params_billion'], gpu['fp16_tflops'], n_concurrent_request, byte_factor)
            generation_time_per_token_per_request = calc_generation_time_per_token(num_gpus, model['params_billion'], gpu['memory_bandwidth_gbps'], n_concurrent_request, byte_factor)
            estimated_ttft = calc_estimated_ttft_time(prefill_time_per_token_per_request, generation_time_per_token_per_request, prompt_size, response_size)
            estimated_response_time = calc_estimated_response_time(prefill_time_per_token_per_request, generation_time_per_token_per_request, prompt_size, response_size)
            if not kv_cache_tokens == "OOM":
                kv_cache_tokens = int(kv_cache_tokens)
            capacity_latency_table.append([model['name'], gpu['name'], num_gpus , f"{kv_cache_tokens}", f"{prefill_time_per_token_per_request:.3f}", f"{generation_time_per_token_per_request:.3f}", f"{estimated_ttft:.3f}", f"{estimated_response_time:.3f}", gpu["vendor"]])

    return capacity_latency_table