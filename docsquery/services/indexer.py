import os
import uuid
import hashlib
import logging

from datetime import datetime
from rich.progress import track

from langchain_chroma import Chroma

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)

logging.getLogger("pypdf").setLevel(logging.ERROR)


def resolve_device(device: str):
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    return device

class EmbeddingsFactory:
    @staticmethod
    def create(config: str):
        provider = config.get("provider", "hf")
        if provider == "hf":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": resolve_device(config.get("device"))},
                encode_kwargs={"batch_size": 32}
            )

        elif provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=config.get("base_url", "http://localhost:11434")
            )

        else:
            raise ValueError(f"Unknown embeddings: {name}")

class DocsLoader:
    def __init__(self, config, folder_path=None, store_path="./chroma_db",
                 chunk_size=500, chunk_overlap=50, collection_name="docs"):
        self.folder_path = folder_path
        self.store_path = store_path
        
        self.embeddings = EmbeddingsFactory.create(config)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.timestamp = str(datetime.now())

        self.collection_name = collection_name
        self.vectordb = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.store_path,
            embedding_function=self.embeddings,
        )

    @staticmethod
    def file_hash(path):
        with open(path, "rb") as f: 
            return hashlib.md5(f.read()).hexdigest()

    def is_already_indexed(self, file_hash):
        results = self.vectordb.get(
            where={"file_hash": file_hash},
            limit=1
        )
        return len(results["ids"]) > 0

    def build_metadata(self, file_path, doc_type):
        return {
            "file": os.path.basename(file_path),
            "file_path": file_path,
            "type": doc_type,
            "folder": self.folder_path,
            "timestamp": self.timestamp,
            "file_hash": self.file_hash(file_path),
            "chunk_id": str(uuid.uuid4()),
        }

    def process_folder(self):
        if not self.folder_path:
            raise ValueError("folder_path is required for indexing")

        all_chunks = []

        for root, _, files in os.walk(self.folder_path):
            for file in track(files, description="Processing files"):
                if file.startswith(("~$", ".", "._")):
                    continue

                path = os.path.join(root, file)
                file_hash = self.file_hash(path)

                if self.is_already_indexed(file_hash):
                    print(f"Skipping (already indexed): {file}")
                    continue

                try:
                    chunks = self.process_file(path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Skipping {file}: {e}")

        if all_chunks:
            self.vectordb.add_documents(all_chunks)

        print(f"Indexed {len(all_chunks)} chunks")

    def process_file(self, file_path):
        ext = file_path.lower().split(".")[-1]

        if ext == "pdf":
            return self.process_pdf(file_path)
        elif ext in ["ppt", "pptx"]:
            return self.process_ppt(file_path)
        elif ext == "docx":
            return self.process_docx(file_path)
        elif ext in ["txt", "md", "log", "notes"]:
            return self.process_text(file_path)

        return []

    def process_docx(self, file_path):
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata.update(self.build_metadata(file_path, "docx"))

        return chunks

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = []

        for page_num, page in enumerate(pages):
            split_chunks = splitter.split_documents([page])

            for chunk in split_chunks:
                metadata = self.build_metadata(file_path, "pdf")
                metadata["page"] = page_num + 1
                chunk.metadata.update(metadata)
                chunks.append(chunk)

        return chunks

    def process_ppt(self, file_path):
        loader = UnstructuredPowerPointLoader(file_path)
        docs = loader.load()

        chunks = []

        for i, doc in enumerate(docs):
            metadata = self.build_metadata(file_path, "ppt")
            metadata["slide"] = i + 1
            doc.metadata.update(metadata)
            chunks.append(doc)

        return chunks

    def process_text(self, file_path):
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        except Exception:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                docs = [Document(page_content=f.read())]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata.update(self.build_metadata(file_path, "text"))

        return chunks

    def get_retriever(self, top_k=3, filters={}):
        search_kwargs={"k": int(top_k)}
        if filters:
            search_kwargs["filter"] = filters
        return self.vectordb.as_retriever(search_kwargs=search_kwargs)
