import os
import warnings
import logging

# Silence warnings
warnings.filterwarnings("ignore")

# Silence torch
logging.getLogger("torch").setLevel(logging.ERROR)

from .services.indexer import DocsLoader
from .services.rag import QA

from .config_manager import ConfigManager

import typer
import requests
from time import perf_counter

app = typer.Typer()

DEFAULT_MODELS = {
    "hf": "google/gemma-3-4b-it",
    "ollama": "llama3.1",
}
DEFAULT_PROVIDER = "hf"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

def pull_ollama_model(base_url: str, model: str):
    typer.echo(f"Pulling model '{model}' from {base_url}...")

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/pull",
            json={"name": model},
        )
        response.raise_for_status()

        typer.echo("✅ Model pulled successfully.")

    except requests.exceptions.RequestException as e:
        typer.echo(f"❌ Failed to pull model: {e}")
        raise typer.Exit(1)

@app.command()
def index(folder: str):
    """Index documents from a provided folder path"""

    typer.echo(f"Indexing folder: {folder}")

    config = ConfigManager().get_config()
    loader = DocsLoader(config, folder_path=folder)
    result = loader.process_folder()

    typer.echo(result)

@app.command()
def setup(
    provider: str = typer.Option("hf", prompt="Provider (hf/ollama)"),
    model: str = typer.Option(None, help="Model name"),
    base_url: str = typer.Option(None, "--url", help="Ollama base URL (only for ollama provider)"),
    device: str = typer.Option(None, help="Device (auto/cpu/cuda)"),
):
    provider = provider.lower().strip()

    if provider not in {"hf", "ollama"}:
        typer.echo("❌ Provider must be 'hf' or 'ollama'")
        raise typer.Exit(1)

    default_model = DEFAULT_MODELS[provider]
    if not model:
        model = typer.prompt("Model", default=default_model)

    config = {
        "provider": provider,
        "model": model,
    }

    if provider == "hf":
        if not device:
            device = typer.prompt("Device (auto/cpu/cuda)", default="auto")

        import torch

        if device == "cuda" and not torch.cuda.is_available():
            typer.echo("⚠️ CUDA not available, falling back to cpu")
            device = "cpu"

        if device == "mps" and not torch.backends.mps.is_available():
            typer.echo("⚠️ MPS not available, falling back to cpu")
            device = "cpu"


    config["device"] = device

    if provider == "ollama" and not base_url:
        base_url = typer.prompt(
            "Ollama URL",
            default=DEFAULT_OLLAMA_URL
        )
        config["base_url"] = base_url

    cfg = ConfigManager()
    cfg.save(config)

    typer.echo(f"Provider: {provider}")
    typer.echo(f"Model: {model}")

    if provider == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import logging as hf_logging
        hf_logging.enable_progress_bar()

        typer.echo(f"Downloading HF model: {model}...")
        AutoTokenizer.from_pretrained(model)
        AutoModelForCausalLM.from_pretrained(model, device_map=config["device"])
        typer.echo("✅ Model cached.")

    elif provider == "ollama":
        typer.echo(f"Ollama URL: {base_url}")

        pull_ollama_model(base_url, model)
        typer.echo("✅ Ollama configured.")

@app.command()
def ask(
    query: str,
    top_k: int = 3,
    folder: str = typer.Option(None, "--folder", "-f"),
    show_sources: bool = typer.Option(
        False,
        "--sources",
        "-s",
        help="Show source documents in the output"
    ),
):
    """Ask a question"""
    filters = {}

    if folder:
        filters["folder"] = folder
    
    config = ConfigManager().get_config()

    loader = DocsLoader(config)
    retriever = loader.get_retriever(top_k=top_k, filters=filters)

    qa = QA(config, retriever)

    start = perf_counter()

    answer, docs = qa.ask(query)

    elapsed_time = round((perf_counter() - start), 2)

    typer.echo(f"\nAnswer ({elapsed_time} s):\n")
    typer.echo(answer)

    if show_sources:
        typer.echo("\nSources:\n")

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            folder = meta.get("folder")

            typer.echo(f"[{i}] {folder}/{meta.get('file', 'unknown')}")

            if "page" in meta:
                typer.echo(f"    page: {meta['page']}")

            if "slide" in meta:
                typer.echo(f"    slide: {meta['slide']}")

    typer.echo()


if __name__ == "__main__":
    app()
