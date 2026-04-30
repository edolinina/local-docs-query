import os
import json
import typer

from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError


class ConfigManager:
    def __init__(self, config_dir=None):
        self.config_dir = config_dir or os.path.expanduser("~/.docsquery")
        self.config_file = os.path.join(self.config_dir, "config.json")

    def _ensure_config_dir(self):
        os.makedirs(self.config_dir, exist_ok=True)

    def validate_config(self, new_config):
        current_config = self.get_config()

        chroma_exists = os.path.exists("./chroma_db")

        if current_config and chroma_exists:
            config_changed = (
                current_config.get("provider") != new_config.get("provider")
                or current_config.get("model") != new_config.get("model")
            )

            if config_changed:
                typer.echo("\n⚠️ Configuration changed!")
                typer.echo("Previously indexed documents were embedded using a different model/provider.")
                typer.echo("👉 You should re-index your documents to avoid incorrect results.")

                if typer.confirm("Do you want to delete existing embeddings now?", default=False):
                    import shutil
                    shutil.rmtree("./chroma_db", ignore_errors=True)
                    typer.echo("🗑️ Existing embeddings removed. Please run `docsquery index` again.")

    def save(self, data):
        self._ensure_config_dir()
        self.validate_config(data)
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if not os.path.exists(self.config_file):
            return None

        with open(self.config_file) as f:
            return json.load(f)

    def require(self):
        config = self.load()

        if not config or "model" not in config:
            typer.echo("❌ Model not configured. Run `docsquery setup` first.")
            raise typer.Exit(code=1)

        return config

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

    def setup_model(self, model_name):
        if not self.validate_model(model_name):
            raise typer.Exit(1)

        self.save({"model": model_name})
        typer.echo(f"✅ Model set to: {model_name}")

    def get_config(self):
        config = self.require()
        return config
