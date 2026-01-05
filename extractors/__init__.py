"""
Indian Identity Document Extractors
Modular OCR extraction logic for different document types.
"""

from .aadhaar import extract_aadhaar
from .driving_license import extract_driving_license
from .pan import extract_pan
from .voter import extract_voter
from .passport import extract_passport

__all__ = [
    'extract_aadhaar',
    'extract_driving_license',
    'extract_pan',
    'extract_voter',
    'extract_passport'
]

