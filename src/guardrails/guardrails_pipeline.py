# src/guardrails/guardrails_pipeline.py
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

def create_guardrails_pipeline(llm, config_path="src/guardrails/rails_config.yaml"):
    config = RailsConfig.from_path(config_path)
    guardrails = RunnableRails(config)
    guarded_llm = guardrails | llm
    return guarded_llm