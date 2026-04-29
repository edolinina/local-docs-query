import os
from transformers import pipeline


class HFLLM:
    def __init__(self, model_name):
        token = os.getenv("HF_TOKEN")
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="cpu",
            dtype="auto",
            token=token,
            trust_remote_code=True
        )

    def generate(self, prompt, temperature=0.2, max_tokens=256):
        result = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        return result[0]["generated_text"]


class QA:
    def __init__(self, retriever, model_name, max_context_chars=4000):
        self.retriever = retriever
        self.llm = HFLLM(model_name)
        self.max_context_chars = max_context_chars

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
If the answer is not in the context, say: "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    def ask(self, query):
        docs = self.retriever.invoke(query)

        if not docs:
            return "No relevant documents found."

        context = self._build_context(docs)
        prompt = self._build_prompt(query, context)

        response = self.llm.generate(prompt)
        return self._clean_response(response, prompt), docs

    def _clean_response(self, response, prompt):
        # remove prompt echo if model returns full text
        if response.startswith(prompt):
            response = response[len(prompt):]

        return response.strip()
