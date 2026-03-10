
import sys
print("Python executable:", sys.executable)
print("sys.path:", sys.path)
from gpt4all import GPT4All
import os
import sys
class MiniOrcaLLM:
    def __init__(self, model_name="orca-mini-3b-gguf2-q4_0.gguf"):
        # GPT4All downloads models to ~/.cache/gpt4all/ by default
        self.model = GPT4All(model_name, allow_download=True)
        
    def invoke(self, inputs):
        prompt = inputs.get("topic", inputs)
        response = self.model.generate(
            prompt, 
            max_tokens=256,
            temp=0.1,  # Lower temperature for more focused reasoning
            top_k=40,
            top_p=0.9
        )
        return response

# Initialize the model
llm = MiniOrcaLLM()
