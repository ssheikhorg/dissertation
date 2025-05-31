import openai
from anthropic import Anthropic
import google.generativeai as genai
from typing import Dict, Any


class ModelClient:
    """Base class for all API model clients"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class OpenAIClient(ModelClient):
    def __init__(self, config):
        super().__init__(config)
        self.client = openai.Client(api_key=config['api_key'])

    def generate(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get('model', 'gpt-4'),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get('temperature', 0.7),
            max_tokens=kwargs.get('max_tokens', 1000)
        )
        return response.choices[0].message.content


class AnthropicClient(ModelClient):
    def __init__(self, config):
        super().__init__(config)
        self.client = Anthropic(api_key=config['api_key'])

    def generate(self, prompt, **kwargs):
        response = self.client.messages.create(
            model=kwargs.get('model', 'claude-3-opus-20240229'),
            max_tokens=kwargs.get('max_tokens', 1000),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class GeminiClient(ModelClient):
    def __init__(self, config):
        super().__init__(config)
        genai.configure(api_key=config['api_key'])
        self.model = genai.GenerativeModel('gemini-pro')

    def generate(self, prompt, **kwargs):
        response = self.model.generate_content(prompt)
        return response.text