"""
Document Type Validation Module
Validates whether OCR text matches the expected document type.
"""

import re
from typing import List


def validate_aadhaar(lines: List[str], text: str) -> bool:
    """
    Validate if the document is an Aadhaar card.
    
    Checks for:
    - Aadhaar number (12 digits)
    - VID (16 digits)
    - Keywords: "AADHAAR", "UIDAI", "Government of India", "Unique Identification"
    - DOB pattern
    - Gender (MALE/FEMALE)
    """
    text_upper = text.upper()
    
    # Check for Aadhaar-specific keywords
    aadhaar_keywords = [
        'AADHAAR', 'UIDAI', 'UNIQUE IDENTIFICATION',
        'GOVERNMENT OF INDIA', 'भारत सरकार'
    ]
    
    keyword_found = any(keyword in text_upper for keyword in aadhaar_keywords)
    
    # Check for Aadhaar number pattern (12 digits)
    aadhaar_pattern = re.search(r'\b\d{4}\s*\d{4}\s*\d{4}\b', text) or re.search(r'\b\d{12}\b', text)
    
    # Check for VID pattern (16 digits)
    vid_pattern = re.search(r'\b\d{4}\s*\d{4}\s*\d{4}\s*\d{4}\b', text) or re.search(r'\b\d{16}\b', text)
    
    # Check for gender
    gender_found = re.search(r'\b(MALE|FEMALE|TRANSGENDER)\b', text_upper)
    
    # Check for DOB pattern
    dob_pattern = re.search(r'\b(DOB|Date of Birth|D\.O\.B)\b', text_upper, re.I) or \
                  re.search(r'\d{2}[/\-]\d{2}[/\-]\d{4}', text)
    
    # Document is valid if it has:
    # - At least one keyword AND (Aadhaar number OR VID)
    # OR
    # - Aadhaar number AND gender
    is_valid = (keyword_found and (aadhaar_pattern or vid_pattern)) or \
               (aadhaar_pattern and gender_found)
    
    return is_valid


def validate_pan(lines: List[str], text: str) -> bool:
    """
    Validate if the document is a PAN card.
    
    Checks for:
    - PAN number format (ABCDE1234F)
    - Keywords: "INCOME TAX", "PERMANENT ACCOUNT NUMBER", "PAN"
    - "Government of India"
    """
    text_upper = text.upper()
    
    # Check for PAN-specific keywords
    pan_keywords = [
        'INCOME TAX', 'PERMANENT ACCOUNT NUMBER',
        'PERMANENT ACCOUNT', 'INCOME TAX DEPARTMENT'
    ]
    
    keyword_found = any(keyword in text_upper for keyword in pan_keywords)
    
    # Check for PAN number pattern (5 letters + 4 digits + 1 letter)
    pan_pattern = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', text)
    
    # Check for "GOVT. OF INDIA" or similar
    govt_india = re.search(r'(GOVERNMENT|GOVT\.?)\s+(OF\s+)?INDIA', text_upper)
    
    # Document is valid if it has:
    # - PAN number pattern AND at least one keyword
    # OR
    # - PAN number AND "Government of India"
    is_valid = (pan_pattern and keyword_found) or \
               (pan_pattern and govt_india)
    
    return is_valid


def validate_driving_license(lines: List[str], text: str) -> bool:
    """
    Validate if the document is a Driving License.
    
    Checks for:
    - Keywords: "DRIVING LICENCE", "DRIVING LICENSE", "DL", "TRANSPORT"
    - DL number pattern
    - COV (Class of Vehicle)
    - Validity dates
    - Blood group
    """
    text_upper = text.upper()
    
    # Check for DL-specific keywords
    dl_keywords = [
        'DRIVING LICEN', 'DRIVING LICENCE', 'DRIVING LICENSE',
        'TRANSPORT', 'MOTOR VEHICLES', 'FORM OF LICENCE'
    ]
    
    keyword_found = any(keyword in text_upper for keyword in dl_keywords)
    
    # Check for DL number patterns (varies by state)
    # Common patterns: MH01 20110012345, DL-1420110012345, etc.
    dl_pattern = re.search(r'\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{11}\b', text) or \
                 re.search(r'\b[A-Z]{2}\d{2}\s?\d{11}\b', text)
    
    # Check for COV (Class of Vehicle)
    cov_found = re.search(r'\b(COV|CLASS OF VEHICLE)\b', text_upper)
    
    # Check for validity
    validity_found = re.search(r'\b(VALIDITY|VALID\s+TILL|VALID\s+UPTO)\b', text_upper, re.I)
    
    # Check for blood group
    blood_group = re.search(r'\b(A\+|A-|B\+|B-|AB\+|AB-|O\+|O-)\b', text)
    
    # Document is valid if it has:
    # - At least one keyword AND (DL pattern OR COV OR validity)
    is_valid = keyword_found and (dl_pattern or cov_found or validity_found)
    
    return is_valid


def validate_voter_id(lines: List[str], text: str) -> bool:
    """
    Validate if the document is a Voter ID card.
    
    Checks for:
    - Keywords: "ELECTION COMMISSION", "ELECTORAL", "VOTER", "EPIC"
    - EPIC number pattern
    - "Elector's Photo Identity Card"
    """
    text_upper = text.upper()
    
    # Check for Voter ID-specific keywords
    voter_keywords = [
        'ELECTION COMMISSION', 'ELECTORAL', 'ELECTOR',
        'VOTER', 'EPIC', 'PHOTO IDENTITY CARD',
        'ELECTORS PHOTO'
    ]
    
    keyword_found = any(keyword in text_upper for keyword in voter_keywords)
    
    # Check for EPIC number pattern (typically 3 letters + 7 digits)
    epic_pattern = re.search(r'\b[A-Z]{3}\d{7}\b', text)
    
    # Check for "Election Commission of India"
    election_commission = re.search(r'ELECTION\s+COMMISSION', text_upper)
    
    # Document is valid if it has:
    # - At least one keyword
    # OR
    # - EPIC pattern AND election commission
    is_valid = keyword_found or (epic_pattern and election_commission)
    
    return is_valid


def validate_passport(lines: List[str], text: str) -> bool:
    """
    Validate if the document is a Passport.
    
    Checks for:
    - Keywords: "PASSPORT", "REPUBLIC OF INDIA", "P<IND"
    - Passport number pattern
    - "Nationality: INDIAN"
    - Place of Birth
    """
    text_upper = text.upper()
    
    # Check for Passport-specific keywords
    passport_keywords = [
        'PASSPORT', 'REPUBLIC OF INDIA',
        'P<IND', 'NATIONALITY', 'PLACE OF BIRTH',
        'DATE OF ISSUE', 'DATE OF EXPIRY'
    ]
    
    keyword_found = any(keyword in text_upper for keyword in passport_keywords)
    
    # Check for passport number pattern (typically 1 letter + 7 digits)
    passport_pattern = re.search(r'\b[A-Z]\d{7}\b', text)
    
    # Check for "Republic of India"
    republic_india = re.search(r'REPUBLIC\s+OF\s+INDIA', text_upper)
    
    # Check for nationality
    nationality = re.search(r'NATIONALITY[:\s]*INDIAN', text_upper)
    
    # Check for MRZ pattern (Machine Readable Zone)
    mrz_pattern = re.search(r'P<IND', text_upper)
    
    # Document is valid if it has:
    # - "PASSPORT" keyword OR "Republic of India"
    # OR
    # - Passport number AND nationality
    # OR
    # - MRZ pattern
    is_valid = ('PASSPORT' in text_upper) or republic_india or \
               (passport_pattern and nationality) or mrz_pattern
    
    return is_valid


def validate_document_type(doc_type: str, lines: List[str], text: str) -> bool:
    """
    Main validation function that routes to specific validators.
    
    Args:
        doc_type: Document type ('aadhaar', 'pan', 'driving_license', 'voter_id', 'passport')
        lines: List of OCR text lines
        text: Full OCR text
        
    Returns:
        True if document matches the expected type, False otherwise
    """
    validators = {
        'aadhaar': validate_aadhaar,
        'pan': validate_pan,
        'driving_license': validate_driving_license,
        'voter_id': validate_voter_id,
        'passport': validate_passport
    }
    
    validator = validators.get(doc_type)
    if not validator:
        return False
    
    return validator(lines, text)
