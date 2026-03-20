
from langchain_core.prompts import PromptTemplate

# --------------------------
# PDF Analysis Prompt Template
# --------------------------
pdf_analysis_prompt = PromptTemplate(
    input_variables=["input", "context"],
    template=(
        "You are given the results of a Bayesian optimization run of a black-box function.\n\n"
        "Context:\n{context}\n\n"
        "Report:\n{input}\n\n"
        "Analyse the run with particular attention to:\n"
        "- where the maximum likely lies\n"
        "- suggest concrete hyperparameter tuning strategies (e.g., adjusting lengthscale, noise term, or kernel type) when appropriate"
        "- how the model evolves across iterations\n"
        "- whether the kernel choice and hyperparameters are appropriate\n"
        "- whether kernel adaptation or a different kernel should be considered\n"
        "\nOnly base your recommendations on the data in the report. Do not invent metrics."
    )
)
