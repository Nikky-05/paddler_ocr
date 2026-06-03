"""
Indian Identity Document Extractors

On this branch the Ollama vision model (``ollama_ocr.py``) is the sole OCR /
extraction engine. The only shared helper still needed at package level is
``split_address``; everything else lives in ``ollama_ocr`` / ``utils``.
"""

from .utils import split_address

__all__ = [
    'split_address',
]
