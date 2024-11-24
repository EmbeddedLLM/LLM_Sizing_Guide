import argparse
from tabulate import tabulate
from transformers import AutoConfig, AutoModel
import csv
import os
import json
import requests
from bs4 import BeautifulSoup

"""
Given the model name
Given the number of concurrent request
Given the context length of model user wants to serve
Search for the minimum number of GPUs each GPU need to serve the targeted model
"""

def perform_calculations(model_list, n_concurrent_request, avg_context_window, datatype):
    print(f" num_gpu = {num_gpu}, prompt_size = {prompt_size} tokens, response_size = {response_size} tokens")
    print(f" n_concurrent_request = {n_concurrent_request}, avg_context_window = {avg_context_window} tokens")
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

    print(f"\n******************** Estimate LLM Memory Footprint ********************")
    # 
    memory_footprint_table = []
    for model_spec in model_specs:
        kv_cache_size_per_token = calc_kv_cache_size_per_token(
            model_spec["n_layers"], 
            model_spec["d_model"], 
            byte_factor
        )
        memory_footprint = calc_memory_footprint(
            model_spec, 
            n_concurrent_request, 
            avg_context_window if avg_context_window > 0 else model_spec["max_context_window"], 
            byte_factor
        )
        for gpu in gpu_specs[datatype]: 
            minimum_num_gpu_required = memory_footprint / gpu['memory_gb']
            memory_footprint_table.append(
                [
                    model_spec['name'], 
                    kv_cache_size_per_token,
                    memory_footprint,
                    round(minimum_num_gpu_required)
                    # f"{kv_cache_size_per_token:.6f}", 
                    # f"{memory_footprint:.2f}"
                ]
            )

    return memory_footprint_table