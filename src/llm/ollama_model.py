# src/llm/ollama_model.py
from langchain_ollama import OllamaLLM

def load_ollama_model(model_name="mistral"):
    """
    Load the local Ollama Mistral model for offline inference.
    """
    return OllamaLLM(model=model_name)