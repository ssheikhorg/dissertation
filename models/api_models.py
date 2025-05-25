# Example integration for multiple models
import openai
from anthropic import Anthropic
import google.generativeai as genai


class OpenAIClient:
    def __init__(self, api_key):
        self.client = openai.Client(api_key)

    def generate(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get('model', 'gpt-4'),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temp', 0.7)
        )
        return response.choices[0].message.content


class AnthropicClient:
    pass
