# Pyvene method of getting activations
import os
import torch
from tqdm import tqdm
import numpy as np
import sys
import argparse
sys.path.append('../')

from transformers import AutoTokenizer, AutoModelForCausalLM

# Specific pyvene imports
from utils import load_task_dataset, get_llama_activations_pyvene
from interveners import wrapper, Collector
from config import get_model_path, INTERVENTION_LAYER, ensure_dirs, FEATURES_DIR
import pyvene as pv

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama3_8B_instruct')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Explicit model path (overrides config lookup)')
    parser.add_argument('--dataset_name', type=str, default='truthfulqa')
    parser.add_argument('--layer', type=int, default=INTERVENTION_LAYER)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = args.model_path or get_model_path(args.model_name)
    ensure_dirs()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"

    dataset = load_task_dataset(args.dataset_name)
    
    print("Tokenizing prompts")
    prompts = []
    labels = []
    
    for item in dataset:
        text = item["text"]
        label = item["label"]
        prompt = tokenizer(text, return_tensors='pt').input_ids
        prompts.append(prompt)
        labels.append(label)

    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for i, prompt in enumerate(tqdm(prompts)):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, device)
        # Extract only the last token activation from the specified layer
        last_token_activation = layer_wise_activations[args.layer, -1, :].copy()  # (D,)
        all_layer_wise_activations.append(last_token_activation)
        
        # For head-wise, also extract only last token
        last_token_head_activation = head_wise_activations[args.layer, -1, :].copy()  # (H*d_head,)
        all_head_wise_activations.append(last_token_head_activation)

    # Convert to numpy arrays
    all_layer_wise_activations = np.array(all_layer_wise_activations, dtype=np.float32)  # (N, D)
    all_head_wise_activations = np.array(all_head_wise_activations, dtype=np.float32)  # (N, H*d_head)
    labels = np.array(labels, dtype=np.int32)  # (N,)

    # Ensure features directory exists
    os.makedirs(FEATURES_DIR, exist_ok=True)

    print("Saving labels")
    np.save(f'{FEATURES_DIR}/{args.model_name}_{args.dataset_name}_labels.npy', labels)

    print("Saving layer wise activations")
    np.save(f'{FEATURES_DIR}/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    np.save(f'{FEATURES_DIR}/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()
