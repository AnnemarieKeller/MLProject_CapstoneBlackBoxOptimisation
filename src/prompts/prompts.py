
from langchain_core.prompts import PromptTemplate

# --------------------------
# PDF Analysis Prompt Template
# --------------------------
pdf_analysis_prompt = PromptTemplate(
    input_variables=["input", "context"],
    template=(
        "You are a data scientist. Analyze the following text from a company's PDF report:\n\n"
        "Context: {context}\n\n"
        "PDF Content: {input}\n\n"
        "Please provide a clear summary, key points, and any relevant observations "
        "or insights. Keep explanations beginner-friendly."
    )
)