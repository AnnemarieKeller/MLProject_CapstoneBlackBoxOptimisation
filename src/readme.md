# PDF Analysis Pipeline

This project provides a pipeline to process PDF reports, split them into chunks, and analyze their contents using local LLMs with Guardrails validation.

---

## Features

- Automatically finds the latest PDF folder and loads all PDFs.
- Extracts text from PDFs and splits them into manageable chunks.
- Processes each chunk through a local LLM with Guardrails schema enforcement.
- Supports multiple LLM configurations:
  - **Ollama Mistral** (offline)
  - **Transformer + LoRA** (optional, fine-tuning) via HuggingFace

---

## Installation

It is recommended to use a virtual environment:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

navigate into the repo src folder via terminal 
```
cd src
```

# Requirements 
Python 3.11 as there are issues with langchain with 3.14

```
pip install -r requirements.txt
```

Install all dependencies via the provided `requirements.txt`. Required packages include:

- `pypdf` for PDF text extraction.
- `langchain_ollama` for local Ollama Mistral LLM support.
- `rich` and `tqdm` for logging and progress visualization.

- Guardrails for structured LLM responses.



## Running the Pipeline

From the `src` folder, run:

```bash
python main.py
```



1. Navigate to the `src` folder.
2. Execute the main script.
3. The pipeline will automatically:
   - Detect the latest PDF folder and its `reports/` subfolder.
   - Extract and chunk PDF text.
   - Process chunks through the selected LLM.
   - Log outputs and errors in real time.
   - Produce structured results using Guardrails.

---



## Configuration

### LLM Selection

There are different models which are being trialed: 

- **Mini-Orca**: Lightweight, small-scale model for rapid inference.
- **Ollama Mistral**: Local Mistral model for offline, high-quality processing.

To switch between models, modify the main script to load the desired model.  

### LoRA Fine-Tuning

Optional LoRA training is  planned to be supported soon :

- Enable by setting the `USE_LORA` flag to true.
- Place LoRA weights in the `src/training/lora_weights` folder.
- The pipeline will then use Transformer + LoRA for inference instead of the default LLM.

---


## Testing PDF Loading

Before running the full pipeline, you can verify PDF extraction and chunking by loading a single PDF and inspecting its chunks. This helps ensure that the pipeline can correctly read and process PDF content.

---

## Logging

- Logs are displayed in the console using rich formatting.
- LangSmith integration provides a structured test logging framework to track outputs for each PDF and chunk.

---

## Notes

- Ensure all dependencies from `requirements.txt` are installed.
- The Guardrails configuration file (`rails_config.yaml`) must match the expected input schema for structured output.
- LoRA training is optional; by default, the pipeline runs Mini-Orca or Ollama Mistral. 
- The system automatically handles dynamic folder paths, including the latest PDF subfolders.

---

## Summary

This pipeline is designed to streamline the analysis of PDF reports using LLMs, offering flexibility with multiple model choices and optional LoRA training. 
