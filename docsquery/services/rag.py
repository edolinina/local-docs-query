import os
import requests


MAX_TOKENS = 256

class BaseLLM:
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

class HFLLM(BaseLLM):
    def __init__(self, model, device):

        from transformers import pipeline
        # Silence transformers logs
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        token = os.getenv("HF_TOKEN")

        self.pipe = pipeline(
            "text-generation",
            model=model,
            device_map=device,
            token=token,
        )

    def generate(self, prompt, max_tokens=MAX_TOKENS):
        result = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        return result[0]["generated_text"]

class OllamaLLM(BaseLLM):
    def __init__(self, model_name, base_url):
        self.model = model_name
        self.url = f"{base_url.rstrip('/')}/api/generate"

    def generate(self, prompt, max_tokens=MAX_TOKENS):
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.2,
                }
            },
        )
        response.raise_for_status()
        return response.json()["response"]

class LLMFactory:
    @staticmethod
    def create(config):
        provider = config["provider"]

        if provider == "hf":
            return HFLLM(config["model"], config.get("device", "auto"))

        elif provider == "ollama":
            return OllamaLLM(
                config["model"],
                config.get("base_url", "http://localhost:11434")
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

class QA:
    def __init__(self, config, retriever, max_context_chars=4000):
        self.retriever = retriever
        self.max_context_chars = max_context_chars
        self.llm = LLMFactory.create(config)

    def _build_context(self, docs):
        context = ""
        for d in docs:
            if len(context) + len(d.page_content) > self.max_context_chars:
                break
            context += d.page_content + "\n\n"
        return context

    def _build_prompt(self, query, context):
        return f"""
You are a helpful assistant.

ONLY use the provided context to answer the question.
If the answer is not in the context, say: "I can't find relevant information in the provided documents".

Context:
{context}

Question:
{query}

Answer:
"""

    def ask(self, query):
        docs = self.retriever.invoke(query)

        if not docs:
            return "No relevant documents found.", []

        context = self._build_context(docs)
        prompt = self._build_prompt(query, context)

        response = self.llm.generate(prompt)
        return self._clean_response(response, prompt), docs

    def _clean_response(self, response, prompt):
        # remove prompt echo if model returns full text
        if response.startswith(prompt):
            response = response[len(prompt):]

        return response.strip()
