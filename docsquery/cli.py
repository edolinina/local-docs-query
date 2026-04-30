import os
import json

import warnings
import logging

# Silence warnings
warnings.filterwarnings("ignore")

# Silence torch
logging.getLogger("torch").setLevel(logging.ERROR)

from .services.indexer import DocsLoader
from .services.rag import QA
from .services.utils import resolve_device, pull_ollama_model

from .config_manager import ConfigManager
from .constants import *

import typer
import requests
from time import perf_counter

app = typer.Typer()


@app.command()
def index(folder: str):
    """Index documents from a provided folder path"""

    typer.echo(f"Indexing folder: {folder}")

    config = ConfigManager().get()
    loader = DocsLoader(config, folder_path=folder)
    result = loader.process_folder()

    typer.echo(result)

@app.command()
def config():
    """Show current configuration"""

    cfg = ConfigManager()
    config = cfg.get()

    if not config:
        typer.echo("❌ No configuration found. Run `docsquery setup` first.")
        raise typer.Exit(1)

    typer.echo("Current configuration:\n")
    typer.echo(json.dumps(config, indent=2))

    typer.echo(f"\nConfig path: {cfg.get_path()}")

@app.command()
def setup(
    embedding_provider: str = typer.Option("hf", prompt="Embedding Provider (hf/ollama)"),
    embedding_model: str = typer.Option(None, help="Embedding model"),
    llm_provider: str = typer.Option("hf", prompt="LLM Provider (hf/ollama)"),
    llm_model: str = typer.Option(None, help="LLM model"),
    base_url: str = typer.Option(None, "--url", help="Ollama base URL"),
    device: str = typer.Option(None, help="Device (auto/cpu/cuda/mps)"),
):

    llm_provider = llm_provider.lower().strip()
    embedding_provider = embedding_provider.lower().strip()

    if embedding_provider not in {"hf", "ollama"}:
        typer.echo("❌ Invalid embedding provider")
        raise typer.Exit(1)

    if llm_provider not in {"hf", "ollama"}:
        typer.echo("❌ Invalid LLM provider")
        raise typer.Exit(1)

    default_emb_model = DEFAULT_EMBEDDINGS[embedding_provider]
    if not embedding_model:
        embedding_model = typer.prompt("Embedding Model", default=default_emb_model)

    default_llm_model = DEFAULT_MODELS[llm_provider]
    if not llm_model:
        llm_model = typer.prompt("LLM Model", default=default_llm_model)

    config = {
        "llm": {
            "provider": llm_provider,
            "model": llm_model,
        },
        "embedding": {
            "provider": embedding_provider,
            "model": embedding_model,
        }
    }

    if embedding_provider == "hf" or llm_provider == "hf":
        if not device:
            device = typer.prompt("Device (auto/cpu/cuda)", default="auto")
        device = resolve_device(device)

        if embedding_provider == "hf":
            config["embedding"]["device"] = device
        if llm_provider == "hf":
            config["llm"]["device"] = device

    if llm_provider == "ollama":
        if not base_url:
            base_url = typer.prompt("Ollama URL", default=DEFAULT_OLLAMA_URL)
        config["llm"]["base_url"] = base_url

    cfg = ConfigManager()
    cfg.save(config)

    typer.echo(f"LLM: {llm_provider} ({llm_model})")
    typer.echo(f"Embeddings: {embedding_provider} ({embedding_model})")

    # Setup models
    if embedding_provider == "ollama":
        cfg.setup_ollama_model(config["llm"]["base_url"], embedding_model)

    if llm_provider == "hf":
        cfg.setup_hf_model(llm_model, device)

    else:
        cfg.setup_ollama_model(config["llm"]["base_url"], llm_model)

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
    
    config = ConfigManager().get()

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
