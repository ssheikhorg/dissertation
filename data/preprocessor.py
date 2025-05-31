import re
import string
from typing import List, Dict


class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text normalization"""
        text = text.lower().strip()
        text = re.sub(f'[{string.punctuation}]', '', text)
        return re.sub(r'\s+', ' ', text)

    @staticmethod
    def preprocess_batch(records: List[Dict]) -> List[Dict]:
        return [{
            'original_prompt': r['prompt'],
            'clean_prompt': TextPreprocessor.clean_text(r['prompt']),
            'original_reference': r.get('reference', ''),
            'clean_reference': TextPreprocessor.clean_text(r.get('reference', ''))
        } for r in records]