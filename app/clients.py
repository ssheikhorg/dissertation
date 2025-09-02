# clients.py - Unified model client with LangChain
import os
from typing import List, Dict, Any, Optional
from langchain.llms import LlamaCpp, HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
import torch

from .config import settings


class MedicalRetriever(BaseRetriever):
    """Medical knowledge retriever for RAG"""

    def __init__(self):
        super().__init__()
        # Medical knowledge base
        self.medical_knowledge = [
            "Diabetes symptoms include increased thirst, frequent urination, fatigue, blurred vision.",
            "Aspirin is a nonsteroidal anti-inflammatory drug that reduces pain and inflammation.",
            "Hypertension (high blood pressure) is a condition where blood pressure is consistently too high.",
            "The liver performs detoxification, protein synthesis, and produces biochemicals for digestion.",
            "COVID-19 symptoms include fever, cough, shortness of breath, fatigue, and loss of taste/smell.",
            "Antibiotics treat bacterial infections but are ineffective against viral infections.",
            "Vaccines stimulate the immune system to produce antibodies against specific diseases.",
            "Cancer treatments include surgery, chemotherapy, radiation therapy, and immunotherapy.",
            "Heart disease risk factors include high blood pressure, high cholesterol, smoking, and diabetes.",
            "Mental health conditions like depression can be treated with therapy and medication."
        ]

        # Create vector store for medical knowledge
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        documents = self.text_splitter.create_documents(self.medical_knowledge)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vectorstore = FAISS.from_documents(documents, embeddings)

    def get_relevant_documents(self, query: str) -> List[str]:
        """Retrieve relevant medical context"""
        docs = self.vectorstore.similarity_search(query, k=3)
        return [doc.page_content for doc in docs]


class UnifiedModelClient:
    """Unified client for all local models using LangChain"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = os.path.join(settings.model_path, model_name)
        self.llm = None
        self.retriever = MedicalRetriever()
        self.chain = None

        self._init_model()
        self._init_chain()

    def _init_model(self):
        """Initialize the local model using LangChain"""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model not found at {self.model_path}")

        try:
            # Try to use LlamaCpp for better performance with GGUF models
            if any(ext in self.model_path.lower() for ext in ['.gguf', '.ggml']):
                self.llm = LlamaCpp(
                    model_path=self.model_path,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                    verbose=False,
                    n_ctx=2048  # Context window size
                )
            else:
                # Use transformers pipeline for other model formats
                tokenizer = AutoTokenizer.from_pretrained(self.model_path)

                # Configure quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )

                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map=settings.device,
                    torch_dtype=torch.float16 if settings.use_fp16 else torch.float32,
                    trust_remote_code=True
                )

                # Create pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=settings.max_tokens,
                    temperature=settings.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

                # Create LangChain LLM
                self.llm = HuggingFacePipeline(pipeline=pipe)

        except Exception as e:
            raise ValueError(f"Failed to initialize model {self.model_name}: {str(e)}")

    def _init_chain(self):
        """Initialize the LangChain with medical context"""
        prompt_template = """You are a medical AI assistant. Use the following medical context to answer the question accurately and factually. 
        If you don't know the answer based on the context, say you don't know. Be concise and avoid speculation.

        Medical Context:
        {context}

        Question: {question}

        Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def generate(self, prompt: str, use_rag: bool = True) -> str:
        """Generate response with optional RAG enhancement"""
        try:
            if use_rag:
                # Retrieve relevant medical context
                context_docs = self.retriever.get_relevant_documents(prompt)
                context = "\n".join(context_docs)

                # Generate with context
                response = self.chain.run(context=context, question=prompt)
            else:
                # Generate without context
                response = self.llm(prompt)

            return response.strip()

        except Exception as e:
            return f"Error generating response: {str(e)}"


class ModelFactory:
    """Factory to create model clients"""

    _model_registry = {
        "claude-3.7": settings.model_path + "/claude-3.7",
        "mistral-7b": settings.model_path + "/mistral-7b",
        "llama-2-7b": "Llama-2-7b-chat-hf",
        "qwen-7b": "Qwen-7B",
        "meditron-7b": "meditron-7b",
        "biomedgpt": "BioMedGPT",
    }

    @classmethod
    def get_client(cls, model_name: str) -> UnifiedModelClient:
        """Get a model client instance"""
        if model_name not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(
                f"Model {model_name} not found. Available models: {available_models}"
            )

        model_dir = cls._model_registry[model_name]
        return UnifiedModelClient(model_dir)

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models"""
        return list(cls._model_registry.keys())