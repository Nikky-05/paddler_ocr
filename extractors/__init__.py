"""
Indian Identity Document Extractors
Modular OCR extraction logic for different document types.
"""

from .aadhaar import extract_aadhaar
from .driving_license import extract_driving_license
from .pan import extract_pan
from .voter import extract_voter
from .passport import extract_passport
from .kyc_report import is_kyc_report, extract_kyc_report
from .utils import mergeOcrIntoResponse, split_address

__all__ = [
    'extract_aadhaar',
    'extract_driving_license',
    'extract_pan',
    'extract_voter',
    'extract_passport',
    'is_kyc_report',
    'extract_kyc_report',
    'mergeOcrIntoResponse',
    'split_address',
]

