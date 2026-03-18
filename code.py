"""
Open Source LLM Access
======================
Three approaches to prompt open source LLMs:
  1. Hugging Face Transformers  — run models locally
  2. Hugging Face Inference API — call models via HF's hosted API
  3. Ollama                     — run models locally via Ollama server
"""

# ── Dependencies ──────────────────────────────────────────────────────────────
# pip install transformers torch accelerate   (approach 1)
# pip install huggingface_hub                 (approach 2)
# pip install ollama                          (approach 3)


# =============================================================================
# APPROACH 1 — Hugging Face Transformers (local inference)
# =============================================================================
def query_local_transformers(
    utterance: str,
    model_name: str = "microsoft/phi-2",   # swap for any HF model ID
    max_new_tokens: int = 256,
) -> str:
    """
    Download and run a model locally using the `transformers` library.
    Good for: full control, no API key needed.
    Requires: a GPU (or patience on CPU) and enough RAM for the model.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,       # use float32 if no GPU
        device_map="auto",               # auto-places layers on GPU/CPU
    )

    inputs = tokenizer(utterance, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


# =============================================================================
# APPROACH 2 — Hugging Face Inference API (hosted, no local GPU needed)
# =============================================================================
def query_hf_inference_api(
    utterance: str,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    hf_token: str = "YOUR_HF_TOKEN",     # get one free at hf.co/settings/tokens
    max_new_tokens: int = 256,
) -> str:
    """
    Call a model hosted on Hugging Face's Inference API.
    Good for: no local setup, easy to swap models.
    Requires: a free Hugging Face account + API token.
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=hf_token)
    response = client.text_generation(
        utterance,
        model=model_name,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    return response


# =============================================================================
# APPROACH 3 — Ollama (local server, easy model management)
# =============================================================================
def query_ollama(
    utterance: str,
    model_name: str = "llama3",          # run `ollama pull llama3` first
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """
    Query a model running in a local Ollama server.
    Good for: the easiest local setup, chat-style interactions.
    Requires: Ollama installed (https://ollama.com) + desired model pulled.
      Install: brew install ollama   /   curl -fsSL https://ollama.com/install.sh | sh
      Pull:    ollama pull llama3
    """
    import ollama

    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": utterance},
        ],
    )
    return response["message"]["content"]


# =============================================================================
# MULTI-MODEL RUNNER — run the same utterance across several models at once
# =============================================================================
def run_on_multiple_models(utterance: str, approach: str = "ollama") -> dict[str, str]:
    """
    Run the same utterance on a list of models and return a results dict.
    approach: "ollama" | "hf_api" | "transformers"
    """
    models = {
        "ollama":       ["llama3", "mistral", "gemma3"],
        "hf_api":       ["mistralai/Mistral-7B-Instruct-v0.3",
                         "google/gemma-7b-it",
                         "meta-llama/Meta-Llama-3-8B-Instruct"],
        "transformers": ["microsoft/phi-2",
                         "google/gemma-2b-it"],
    }

    dispatch = {
        "ollama":       query_ollama,
        "hf_api":       query_hf_inference_api,
        "transformers": query_local_transformers,
    }

    fn = dispatch[approach]
    results = {}
    for model in models[approach]:
        print(f"\n▶ Querying {model}...")
        try:
            results[model] = fn(utterance, model_name=model)
        except Exception as e:
            results[model] = f"[ERROR] {e}"
    return results


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    UTTERANCE = "Te dieron trabajo"

    print("Connecting to Ollama...")
    try:
        import ollama
        # Check what models are available
        models = ollama.list()
        print(f"Available models: {models}")

        print(f"\nSending utterance: {UTTERANCE}\n")
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are texting someone back."},
                {"role": "user",   "content": UTTERANCE},
            ],
        )
        print("Response:")
        print(response["message"]["content"])

    except Exception as e:
        print(f"ERROR: {e}")