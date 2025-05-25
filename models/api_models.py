# Example integration for multiple models
import openai
from anthropic import Anthropic
import google.generativeai as genai


class ModelEvaluator:
    def __init__(self):
        self.clients = {
            'gpt': openai.Client(),
            'claude': Anthropic(),
            'gemini': genai.configure(api_key='...')
        }

    def query_model(self, model_name, prompt):
        if model_name == 'gpt':
            response = self.clients['gpt'].chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        elif model_name == 'claude':
            pass
    # Similar for other models
