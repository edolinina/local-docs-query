import os
import warnings
import logging

# Silence warnings
warnings.filterwarnings("ignore")

# Silence transformers logs
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# Silence torch
logging.getLogger("torch").setLevel(logging.ERROR)

from .services.indexer import DocsLoader
from .services.rag import QA

from .config_manager import ConfigManager

import typer
from time import perf_counter

app = typer.Typer(add_completion=False)

DEFAULT_MODEL = "google/gemma-3-4b-it" #"microsoft/Phi-3-mini-4k-instruct"

@app.command()
def index(folder):
    """Index documents from a provided folder path"""

    typer.echo(f"Indexing folder: {folder}")

    loader = DocsLoader(folder_path=folder)
    result = loader.process_folder()

    typer.echo(result)

@app.command()
def setup(model: str = typer.Option(
        DEFAULT_MODEL,
        prompt=True,
        help="Hugging Face model name"
    )):
    """Configure the HuggingFace LLM model and pre-download it"""
    hf_logging.enable_progress_bar()

    cfg = ConfigManager()
    cfg.setup_model(model)
    
    typer.echo(f"Model set to: {model}")

    typer.echo(f"Downloading {model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoTokenizer.from_pretrained(model)
    AutoModelForCausalLM.from_pretrained(model)

    typer.echo("Model downloaded and cached.")


@app.command()
def ask(query, top_k=5):
    """Ask a question"""
    cfg = ConfigManager()
    model = cfg.get_model()

    loader = DocsLoader()
    retriever = loader.get_retriever(top_k=top_k)

    qa = QA(retriever, model_name=model)

    start = perf_counter()

    answer, docs = qa.ask(query)

    elapsed_time = round((perf_counter() - start), 2)

    typer.echo(f"\nAnswer ({elapsed_time} s):\n")
    typer.echo(answer)

    typer.echo("\nSources:\n")

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata

        typer.echo(f"[{i}] {meta.get('file', 'unknown')}")

        if "page" in meta:
            typer.echo(f"    page: {meta['page']}")

        if "slide" in meta:
            typer.echo(f"    slide: {meta['slide']}")

        typer.echo()


if __name__ == "__main__":
    app()
