#!/bin/bash

# Define arrays for different parameters
num_gpus=(1 2 4 8)
n_concurrent_reqs=(1 2 4 8 16 32 64 128)
ctx_windows=(4096 8192 16384 32768)

# Define use cases with their respective prompt and response sizes
declare -A use_cases=(
    ["Multi-turn_Chat"]="1000,4000 200"
    ["Summarization"]="2000,8000 100,500"
    ["Literature_Creation"]="500,2000 1000,10000"
    ["RAG"]="2000,6000 100,1000"
    ["Agentic_AI"]="1000,5000 500,2000"
    ["Code_Generation"]="500,3000 100,2000"
    ["Translation"]="100,2000 100,2000"
    ["Question_Answering"]="500,3000 50,500"
)

# Function to run the Python script with given parameters
run_benchmark() {
    local use_case=$1
    local num_gpu=$2
    local prompt_sz=$3
    local response_sz=$4
    local n_concurrent_req=$5
    local ctx_window=$6

    echo "Running benchmark for $use_case with $num_gpu GPUs, prompt size $prompt_sz, response size $response_sz, $n_concurrent_req concurrent requests, and context window $ctx_window"
    HF_TOKEN=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 python LLM_size_pef_calculator_hf_autoconfig.py \
        --num_gpu $num_gpu \
        --prompt_sz $prompt_sz \
        --response_sz $response_sz \
        --n_concurrent_req $n_concurrent_req \
        --ctx_window $ctx_window \
        --use_case "$use_case"
}

# Iterate through all combinations
for use_case in "${!use_cases[@]}"; do
    IFS=' ' read -r prompt_sizes response_sizes <<< "${use_cases[$use_case]}"
    IFS=',' read -ra prompt_array <<< "$prompt_sizes"
    IFS=',' read -ra response_array <<< "$response_sizes"

    for num_gpu in "${num_gpus[@]}"; do
        for prompt_sz in "${prompt_array[@]}"; do
            for response_sz in "${response_array[@]}"; do
                for n_concurrent_req in "${n_concurrent_reqs[@]}"; do
                    for ctx_window in "${ctx_windows[@]}"; do
                        run_benchmark "$use_case" $num_gpu $prompt_sz $response_sz $n_concurrent_req $ctx_window &
                    done
                done
            done
        done
    done
done