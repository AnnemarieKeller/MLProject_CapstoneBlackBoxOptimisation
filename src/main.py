
# # #
import os
from utils.logging import log_info, log_error, LangSmithTestHandler
from utils.utils import get_latest_pdf_folder, get_pdf_paths_from_folder
from ingestion.pdf_loader import load_and_chunk_pdf
from guardrails.guardrails_pipeline import create_guardrails_pipeline
from prompts.prompts import pdf_analysis_prompt
from llm.ollama_model import load_ollama_model

USE_LORA = False # True = Transformer + LoRA, False = Mini-Orca only

# LangSmith test logging
langsmith_handler = LangSmithTestHandler(project_name="pdf_analysis_test")

# --------------------------
# Load LLM
# --------------------------
try:
    if USE_LORA:
        from llm.transformer_model import load_transformer_model
        lora_weights_path = "src/training/lora_weights"
        model, tokenizer = load_transformer_model(lora_path=lora_weights_path)
        log_info("Loaded Transformer model with LoRA.")
    else:
        # from llm.mini_orca_model import load_mini_orca
        # model = load_mini_orca("models/mini-orca-small.gguf")
         model = load_ollama_model()
        # #log_info("Ollama Mistral loaded locally.")
         log_info("Loaded local model.")
except Exception as e:
    log_error(f"Failed to load LLM: {e}")
    raise
 
# --------------------------
# Guardrails pipeline
# --------------------------
try:
    guarded_llm = create_guardrails_pipeline(
        model,
        config_path="src/guardrails/rails_config.yaml"
    )
    log_info("Guardrails pipeline initialized.")
except Exception as e:
    log_error(f"Failed to initialize Guardrails: {e}")
    raise

# --------------------------
# Load PDFs
# --------------------------
try:
    latest_folder = get_latest_pdf_folder()
    pdf_paths = get_pdf_paths_from_folder(latest_folder)
    log_info(f"Found {len(pdf_paths)} PDFs in folder: {latest_folder}")
except Exception as e:
    log_error(f"Error loading PDF paths: {e}")
    raise

# --------------------------
# Process PDFs
# --------------------------
for pdf_path in pdf_paths:
    try:
        log_info(f"Processing PDF: {pdf_path}")
        chunks = load_and_chunk_pdf(pdf_path)

        for chunk in chunks:
            context = "Company follows IFRS accounting, historical trends stable."
            input_text = pdf_analysis_prompt.format(input=chunk.page_content, context=context)

            if USE_LORA:
                inputs = tokenizer(input_text, return_tensors="pt").to("cpu")  # CPU-safe
                output_ids = model.generate(**inputs, max_new_tokens=256)
                output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                output = guarded_llm.invoke({"input": input_text})

            log_info(f"LLM output: {output}")
            langsmith_handler.on_llm_end(output)

    except Exception as e:
        log_error(f"Error processing {pdf_path}: {e}")
