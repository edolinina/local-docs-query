import os
import json
import typer
import shutil
import requests

from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError


class ConfigManager:
    def __init__(self, config_dir=None):
        self.config_dir = config_dir or os.path.expanduser("~/.docsquery")
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.data = self.load()

    def _ensure_config_dir(self):
        os.makedirs(self.config_dir, exist_ok=True)

    def get_path(self):
        return self.config_file

    def invalidate_db(self, new_config):
        current_config = self.data
        if not current_config:
            return

        embedding = current_config["embedding"] \
            if "embedding" in current_config else current_config

        vector_store = current_config.get("vector_store")
        vector_store_exists = os.path.exists(vector_store)

        if embedding and vector_store_exists:
            config_changed = (
                embedding.get("provider") != new_config["embedding"]["provider"]
                or embedding.get("model") != new_config["embedding"]["model"]
            )

            if config_changed:
                typer.echo("\n⚠️ Embedding configuration changed!")
                typer.echo("Previously indexed documents were embedded using a different model/provider.")
                typer.echo("👉 You should re-index your documents to avoid incorrect results.")

                if typer.confirm("Do you want to delete existing embeddings now?", default=False):
                    shutil.rmtree(vector_store, ignore_errors=True)
                    typer.echo("🗑️ Existing embeddings removed. Please run `docsquery index` again.")

    def save(self, data):
        self._ensure_config_dir()
        self.invalidate_db(data)
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not os.path.exists(self.config_file):
            return {}

        with open(self.config_file) as f:
            return json.load(f)

    def get(self):
        self.require()
        return self.data

    def require(self):
        if not self.get_embedding_model():
            typer.echo("❌ Embedding model not configured. Run `docsquery setup` first.")
            raise typer.Exit(code=1)

        if not self.get_llm_model():
            typer.echo("❌ LLM model not configured. Run `docsquery setup` first.")
            raise typer.Exit(code=1)

    def validate_model(self, model_name):
        try:
            model_info(
                model_name,
                token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            )
            return True

        except RepositoryNotFoundError:
            typer.echo(f"❌ Model '{model_name}' not found on Hugging Face.")

        except GatedRepoError:
            typer.echo(
                f"❌ Model '{model_name}' is gated. Accept license and run `hf auth login`."
            )

        except Exception:
            typer.echo(
                f"❌ Failed to access model '{model_name}'. Check internet or token."
            )

        return False

    def setup_hf_model(self, model, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import logging as hf_logging

        hf_logging.enable_progress_bar()

        print(f"Downloading HF model: {model}...")
        AutoTokenizer.from_pretrained(model)
        AutoModelForCausalLM.from_pretrained(model)
        print("✅ Model cached.")

    def setup_ollama_model(self, base_url, model):
        from .services.utils import pull_ollama_model
  
        typer.echo(f"Pulling model '{model}' from {base_url}...")
        try:
            pull_ollama_model(base_url, model)
            typer.echo("✅ Model pulled successfully.")

        except requests.exceptions.RequestException as e:
            typer.echo(f"❌ Failed to pull model: {e}")
            raise typer.Exit(1)

    def get_embedding_model(self):
        return self.data.get("embedding", {}).get("model")

    def get_llm_model(self):
        return self.data.get("llm", {}).get("model")
