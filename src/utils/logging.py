# src/utils/logs.py
import logging

# --------------------------
# Basic Logging Setup
# --------------------------
logger = logging.getLogger("pdf_pipeline")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --------------------------
# Logging Functions
# --------------------------
def log_info(message: str):
    logger.info(message)

def log_error(message: str):
    logger.error(message)


# src/utils/logging.py

class LangSmithTestHandler:
    """
    Simple handler to log LLM outputs for testing.
    No inheritance needed with LangChain 0.3.x.
    """
    def __init__(self, project_name="pdf_analysis_test"):
        self.project_name = project_name

    def on_llm_start(self, prompts):
        print(f"[LangSmith] LLM started with prompts: {prompts}")

    def on_llm_new_token(self, token):
        # optional streaming of tokens
        print(token, end="")

    def on_llm_end(self, output):
        print(f"\n[LangSmith] LLM finished. Output:\n{output}")