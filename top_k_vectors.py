# -*- coding: utf-8 -*-
!pip install neuronpedia

import os, json, random, io, zipfile, requests, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import yaml
from neuronpedia.np_sae_feature import SAEFeature
from google.colab import userdata

# Set Neuronpedia API key from Colab secrets
neuron_key_real = userdata.get('NEURON_KEY')
os.environ["NEURONPEDIA_API_KEY"] = neuron_key_real

# Check for Neuronpedia API key
if "NEURONPEDIA_API_KEY" not in os.environ:
    raise RuntimeError(
        "No Neuronpedia API key found.\n"
        "Please set it before running this cell, for example:\n"
        "import os\n"
        "os.environ['NEURONPEDIA_API_KEY'] = 'your_api_key_here'\n"
        "You can create/get your key from https://neuronpedia.org (Account settings)."
    )

MODEL_ID = "gemma-2-2b"

# SAE Class
class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon

# Hook to grab residual activations
def gather_residual_activations(model, target_layer, inputs):
    target_act = None
    def hook(mod, inputs, outputs):
        nonlocal target_act
        target_act = outputs[0]
        return outputs

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        handle = model.model.layers[target_layer].register_forward_hook(hook)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        handle = model.transformer.h[target_layer].register_forward_hook(hook)
    else:
        raise AttributeError("Model structure not recognized")

    _ = model(inputs)
    handle.remove()
    return target_act

# Dataset Download
def prepare_piqa_files():
    url = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    with zf.open("physicaliqa-train-dev/dev.jsonl") as fin, open("dev.jsonl", "wb") as fout:
        fout.write(fin.read())
    with zf.open("physicaliqa-train-dev/dev-labels.lst") as fin, open("dev-labels.lst", "wb") as fout:
        fout.write(fin.read())
    print("PIQA dev set downloaded.")

# Prediction + Activation Collection
def get_gemma_prediction_and_activations(model, tokenizer, sae_target_layer, question, sol1, sol2, device):
    prompt = f"Question: {question}\nSolution A: {sol1}\nSolution B: {sol2}\nWhich is the better solution? Solution "
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(inputs).logits[0, -1, :]

    token_id_A = tokenizer.encode("A", add_special_tokens=False)[0]
    token_id_B = tokenizer.encode("B", add_special_tokens=False)[0]
    pred_idx = 0 if logits[token_id_A] > logits[token_id_B] else 1

    try:
        acts = gather_residual_activations(model, sae_target_layer, inputs)
    except:
        acts = None
    return pred_idx, acts

# Fetch feature explanations
def fetch_explanation(layer_idx, feature_id):
    sae_name = f"{layer_idx}-gemmascope-res-16k"
    try:
        sae_feature = SAEFeature.get(MODEL_ID, sae_name, str(feature_id))
        data = json.loads(sae_feature.jsonData)
        explanation = data.get("explanations", [{}])[0].get("description", "Explanation not found.")
        return explanation
    except Exception as e:
        return f"Could not fetch L{layer_idx} F{feature_id}: {e}"

# Main
def main():
    if not (os.path.exists("dev.jsonl") and os.path.exists("dev-labels.lst")):
        prepare_piqa_files()

    data = [json.loads(line) for line in open("dev.jsonl", "r", encoding="utf-8")]
    labels = [int(l.strip()) for l in open("dev-labels.lst", "r", encoding="utf-8")]

    sampled_indices = random.sample(range(len(data)), 20)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/gemma-2-2b"
    print("Loading Gemma-2B...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct_ids, incorrect_ids = [], []
    for idx in sampled_indices:
        ex = data[idx]
        pred, _ = get_gemma_prediction_and_activations(
            model, tokenizer, 0, ex["goal"], ex["sol1"], ex["sol2"], device
        )
        if pred == labels[idx] and len(correct_ids) < 5:
            correct_ids.append(idx)
        elif pred != labels[idx] and len(incorrect_ids) < 5:
            incorrect_ids.append(idx)
        if len(correct_ids) == 5 and len(incorrect_ids) == 5:
            break

    print(f"Selected {len(correct_ids)} correct and {len(incorrect_ids)} incorrect examples out of 20 sampled.")

    chosen_examples = {"correct": [], "incorrect": []}

    for idx in correct_ids:
        ex = data[idx]
        label = labels[idx]
        chosen_examples["correct"].append({
            "goal": ex["goal"],
            "sol1": ex["sol1"],
            "sol2": ex["sol2"],
            "correct_answer": "A" if label == 0 else "B"
        })

    for idx in incorrect_ids:
        ex = data[idx]
        label = labels[idx]
        chosen_examples["incorrect"].append({
            "goal": ex["goal"],
            "sol1": ex["sol1"],
            "sol2": ex["sol2"],
            "correct_answer": "A" if label == 0 else "B"
        })

    with open("chosen_examples.json", "w") as f:
        json.dump(chosen_examples, f, indent=2)

    print("Saved chosen examples to chosen_examples.json")

    # Feature explanations for top-k features
    cosine_results = {
        5:  {"top_k_features": [14908, 11854, 13789, 417, 6383],
             "top_k_similarities": [0.2721, 0.2502, 0.1753, 0.1434, 0.1321]},
        10: {"top_k_features": [3031, 11295, 13346, 15191, 15675],
             "top_k_similarities": [0.2316, 0.1470, 0.1361, 0.1235, 0.1218]},
        15: {"top_k_features": [7818, 10716, 3556, 8134, 903],
             "top_k_similarities": [0.2518, 0.2255, 0.1484, 0.1443, 0.1362]},
        20: {"top_k_features": [5121, 6631, 14944, 7854, 3749],
             "top_k_similarities": [0.2542, 0.2369, 0.2126, 0.1556, 0.1375]}
    }

    all_explanations = {}
    for layer, info in cosine_results.items():
        all_explanations[layer] = {}
        for feat_id in info["top_k_features"]:
            explanation = fetch_explanation(layer, feat_id)
            all_explanations[layer][feat_id] = explanation
            time.sleep(0.1)

    with open("feature_explanations.json", "w") as f:
        json.dump(all_explanations, f, indent=2)

    print("Saved feature explanations to feature_explanations.json")

    # Merge cosine similarities with explanations
    merged = {}
    for layer, info in cosine_results.items():
        merged[str(layer)] = {}
        for feat_id, cos in zip(info["top_k_features"], info["top_k_similarities"]):
            explanation = all_explanations.get(layer, {}).get(feat_id, "Explanation not found")
            merged[str(layer)][str(feat_id)] = {
                "cosine_similarity": cos,
                "explanation": explanation
            }

    with open("features_with_cosines.json", "w") as f:
        json.dump(merged, f, indent=2)

    print("Merged file saved to features_with_cosines.json")

if __name__ == "__main__":
    main()
