import argparse
from tabulate import tabulate
from transformers import AutoConfig
import csv
import os

MODEL_LIST=[
    ("meta-llama/Llama-3.1-8B-Instruct", 8.03),
    ("meta-llama/Llama-3.1-70B-Instruct", 70.6),
    ("meta-llama/Llama-3.1-405B-Instruct", 406),
    ("Qwen/Qwen2.5-3B-Instruct", 3),
    ("Qwen/Qwen2.5-7B-Instruct", 7),
    ("Qwen/Qwen2.5-14B-Instruct", 14),
    ("Qwen/Qwen2.5-32B-Instruct", 32),
    ("Qwen/Qwen2.5-72B-Instruct", 72),
    ("Qwen/Qwen2.5-72B-Instruct", 72),
    ("mistralai/Ministral-8B-Instruct-2410", 8.02),
    ("mistralai/Mistral-Nemo-Base-2407", 12.2),
    ("mistralai/Mixtral-8x22B-v0.1", 141),
    ("mistralai/Mistral-7B-Instruct-v0.3", 7.25),

]

model_specs = []

OUTPUT_DIR = 'benchmark'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def count_parameters(config):
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    max_position_embeddings = config.max_position_embeddings

    # Embeddings
    embedding_params = vocab_size * hidden_size + max_position_embeddings * hidden_size

    # Transformer layers
    attention_params = 4 * hidden_size * hidden_size * num_hidden_layers
    ffn_params = 2 * hidden_size * intermediate_size * num_hidden_layers

    # Output layer
    output_params = hidden_size * vocab_size

    total_params = embedding_params + attention_params + ffn_params + output_params
    return total_params

def save_to_csv(data, headers, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description='LLM Performance Calculator')
    parser.add_argument('-g', '--num_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('-p', '--prompt_sz', type=int, default=4096, help='Prompt size in tokens')
    parser.add_argument('-r', '--response_sz', type=int, default=256, help='Response size in tokens')
    parser.add_argument('-c', '--n_concurrent_req', type=int, default=10, help='Number of concurrent requests')
    parser.add_argument('-w', '-cw', '--ctx_window', type=int, default=1024, help='Average context window')
    parser.add_argument('-u', '--use_case', type=str, default='General', help='Use case for the benchmark')

    args = parser.parse_args()

    num_gpu = args.num_gpu
    prompt_size = args.prompt_sz
    response_size = args.response_sz
    n_concurrent_request = args.n_concurrent_req
    avg_context_window = args.ctx_window
    use_case = args.use_case

    # Print input
    print(f" Use case: {use_case}")
    print(f" num_gpu = {num_gpu}, prompt_size = {prompt_size} tokens, response_size = {response_size} tokens")
    print(f" n_concurrent_request = {n_concurrent_request}, avg_context_window = {avg_context_window} tokens")


    # Define variables
    gpu_specs = [
        {"name": "A10", "fp16_tflops": 125, "memory_gb": 24, "memory_bandwidth_gbps": 600},
        {"name": "L20", "fp16_tflops": 59.35, "memory_gb": 48, "memory_bandwidth_gbps": 864},
        {"name": "L40", "fp16_tflops": 181, "memory_gb": 48, "memory_bandwidth_gbps": 864},
        {"name": "L40s", "fp16_tflops": 362, "memory_gb": 48, "memory_bandwidth_gbps": 864},
        {"name": "A100 40 GB", "fp16_tflops": 312, "memory_gb": 40, "memory_bandwidth_gbps": 1555},
        {"name": "A100 40 GB SXM", "fp16_tflops": 312, "memory_gb": 40, "memory_bandwidth_gbps": 1555},
        {"name": "A100 80 GB PCIe", "fp16_tflops": 312, "memory_gb": 80, "memory_bandwidth_gbps": 1935},
        {"name": "A100 80 GB SXM", "fp16_tflops": 312, "memory_gb": 80, "memory_bandwidth_gbps": 2039},
        {"name": "H100 PCIe", "fp16_tflops": 756, "memory_gb": 80, "memory_bandwidth_gbps": 2000},
        {"name": "H100 SXM", "fp16_tflops": 989, "memory_gb": 80, "memory_bandwidth_gbps": 3350},
        {"name": "H100 SXM", "fp16_tflops": 989, "memory_gb": 80, "memory_bandwidth_gbps": 3350},
        {"name": "Intel Gaudi 2", "fp16_tflops": 432, "memory_gb": 96, "memory_bandwidth_gbps": 2460},
        {"name": "Intel Gaudi 3", "fp16_tflops": 1835, "memory_gb": 128, "memory_bandwidth_gbps": 3700},
        {"name": "AMD MI210", "fp16_tflops": 181, "memory_gb": 64, "memory_bandwidth_gbps": 1600},
        # {"name": "AMD MI250", "fp16_tflops": 362, "memory_gb": 128, "memory_bandwidth_gbps": 3280},
        # {"name": "AMD MI250X", "fp16_tflops": 383, "memory_gb": 128, "memory_bandwidth_gbps": 3280},
        {"name": "AMD MI300X", "fp16_tflops": 1300, "memory_gb": 192, "memory_bandwidth_gbps": 5300},
        {"name": "AMD MI325X", "fp16_tflops": 1307.4, "memory_gb": 256, "memory_bandwidth_gbps": 6000},
        {"name": "AMD H200 SXM", "fp16_tflops": 989, "memory_gb": 141, "memory_bandwidth_gbps": 4800},
        {"name": "Google TPU v4", "fp16_tflops": 275, "memory_gb": 32, "memory_bandwidth_gbps": 1200},
        {"name": "Google TPU v5e", "fp16_tflops": 197, "memory_gb": 16, "memory_bandwidth_gbps": 1600},
        {"name": "Google TPU v5p", "fp16_tflops": 459, "memory_gb": 95, "memory_bandwidth_gbps": 4800},
        # Add or comment out GPU types as needed
    ]

    for model, model_params_billion in MODEL_LIST:
        model_config = AutoConfig.from_pretrained(model)

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
            "params_billion": model_params_billion,
            "d_model": model_config.hidden_size,
            "n_heads": model_config.num_attention_heads,
            "n_layers": model_config.num_hidden_layers,
            "max_context_window": context_length,
            "d_head": model_config.hidden_size // model_config.num_attention_heads,
        })
        print(model_specs[-1])


    BYTES_IN_GB = 1_073_741_824  # 1 GB = 1,073,741,824 bytes

    def calc_kv_cache_size_per_token(n_layers, d_model):
        return 2 * 2 * n_layers * d_model / BYTES_IN_GB  # GB/token

    def calc_memory_footprint(model_spec, n_concurrent_request, avg_context_window):
        kv_cache_size_per_token = calc_kv_cache_size_per_token(model_spec["n_layers"], model_spec["d_model"])
        target_gpu_mem = kv_cache_size_per_token * avg_context_window * n_concurrent_request + model_spec["params_billion"] * 2
        return target_gpu_mem

    print(f"\n******************** Estimate LLM Memory Footprint ********************")
    memory_footprint_table = []
    for model_spec in model_specs:
        kv_cache_size_per_token = calc_kv_cache_size_per_token(model_spec["n_layers"], model_spec["d_model"])
        memory_footprint = calc_memory_footprint(model_spec, n_concurrent_request, avg_context_window)
        memory_footprint_table.append([model_spec['name'], f"{kv_cache_size_per_token:.6f}", f"{memory_footprint:.2f}"])
    print(tabulate(memory_footprint_table, headers=['Model', 'KV Cache Size per Token (GiB/token)', 'Memory Footprint (GB)'], tablefmt='orgtbl'))
    # Save memory footprint table
    # memory_footprint_filename = f"memory_footprint_gpu{num_gpu}_prompt{prompt_size}_response{response_size}_concurrent{n_concurrent_request}_context{avg_context_window}.csv"
    memory_footprint_filename = f"memory_footprint_{use_case}_gpu{num_gpu}_prompt{prompt_size}_response{response_size}_concurrent{n_concurrent_request}_context{avg_context_window}.csv"
    save_to_csv(memory_footprint_table, ['Model', 'KV Cache Size per Token (GiB/token)', 'Memory Footprint (GB)'], os.path.join(OUTPUT_DIR, memory_footprint_filename))
    print(f"Memory footprint data saved to {memory_footprint_filename}")

    def calc_kv_cache_tokens(num_gpu, gpu_memory_gb, model_params_billion, kv_cache_size):
        result = (num_gpu * gpu_memory_gb - 2 * model_params_billion) / kv_cache_size
        return result if result >= 0 else "OOM"

    def calc_prefill_time_per_token(num_gpu, model_params_billion, fp16_tflops):
        result = (2 * model_params_billion / num_gpu) / fp16_tflops
        return result if result >= 0 else "OOM"

    def calc_generation_time_per_token(num_gpu, model_params_billion, memory_bandwidth_gbps):
        result = (2 * model_params_billion / num_gpu) / memory_bandwidth_gbps * 1000
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
        # print(f"Model: {model['name']} ({model['params_billion']}B parameters)")
        kv_cache_size = calc_kv_cache_size_per_token(model['n_layers'], model['d_model'])
        for gpu in gpu_specs:
            kv_cache_tokens = calc_kv_cache_tokens(num_gpu, gpu['memory_gb'], model['params_billion'], kv_cache_size)
            prefill_time_per_token = calc_prefill_time_per_token(num_gpu, model['params_billion'], gpu['fp16_tflops'])
            generation_time_per_token = calc_generation_time_per_token(num_gpu, model['params_billion'], gpu['memory_bandwidth_gbps'])
            estimated_ttft = calc_estimated_ttft_time(prefill_time_per_token, generation_time_per_token, prompt_size, response_size)
            estimated_response_time = calc_estimated_response_time(prefill_time_per_token, generation_time_per_token, prompt_size, response_size)
            capacity_latency_table.append([model['name'], gpu['name'], f"{kv_cache_tokens}", f"{prefill_time_per_token:.3f}", f"{generation_time_per_token:.3f}", f"{estimated_ttft:.3f}", f"{estimated_response_time:.3f}"])
    print(tabulate(capacity_latency_table, headers=['Model', 'GPU', 'KV Cache Tokens', 'Prefill Time (ms)', 'Generation Time (ms)', 'Estimated Time To First Token (TTFT) (s)', 'Estimated Response Time (s)'], tablefmt='orgtbl'))
    # Save capacity and latency table
    # capacity_latency_filename = f"capacity_latency_gpu{num_gpu}_prompt{prompt_size}_response{response_size}_concurrent{n_concurrent_request}_context{avg_context_window}.csv"
    capacity_latency_filename = f"capacity_latency_{use_case}_gpu{num_gpu}_prompt{prompt_size}_response{response_size}_concurrent{n_concurrent_request}_context{avg_context_window}.csv"
    save_to_csv(capacity_latency_table, ['Model', 'GPU', 'KV Cache Tokens', 'Prefill Time (ms)', 'Generation Time (ms)', 'Estimated Time To First Token (TTFT) (s)', 'Estimated Response Time (s)'], os.path.join(OUTPUT_DIR, capacity_latency_filename))
    print(f"Capacity and latency data saved to {capacity_latency_filename}")


if __name__ == '__main__':
    main()