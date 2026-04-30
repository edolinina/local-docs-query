import requests

def resolve_device(device: str):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"

    return device

def pull_ollama_model(base_url: str, model: str):
    response = requests.post(
        f"{base_url.rstrip('/')}/api/pull",
        json={"name": model},
    )
    response.raise_for_status()
