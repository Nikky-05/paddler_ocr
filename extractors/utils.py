import re


def validate_verhoeff(number: str) -> bool:
    """Validate 12-digit Aadhaar number using Verhoeff algorithm."""
    if not number or len(number) != 12 or not number.isdigit():
        return False

    # Verhoeff tables
    d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    p = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]

    c = 0
    # Process from right to left (Verhoeff is sensitive to position)
    for i, num in enumerate(number[::-1]):
        c = d[c][p[i % 8][int(num)]]

    return c == 0


def extract_english_only(s: str) -> str:
    """Extract only English/ASCII words from a potentially mixed-script string."""
    if not s:
        return ""
    # Keep only ASCII characters and spaces
    ascii_only = "".join(c if ord(c) < 128 else " " for c in s)
    # Clean up multiple spaces
    return re.sub(r'\s+', ' ', ascii_only).strip()


INDIAN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "orissa", "punjab", "rajasthan",
    "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "new delhi", "chandigarh",
    "puducherry", "pondicherry", "jammu and kashmir", "jammu & kashmir",
    "ladakh", "andaman and nicobar", "dadra and nagar haveli",
    "daman and diu", "lakshadweep",
]


def split_address(address_str: str) -> dict:
    """Split an Indian address string into structured components.

    Returns a dict with keys: building, city, district, pin, floor, house,
    locality, state, street, complex, landmark, untagged.
    All values are strings (empty string when not detected).
    Pure function - never raises.
    """
    result = {
        "building": "",
        "city": "",
        "district": "",
        "pin": "",
        "floor": "",
        "house": "",
        "locality": "",
        "state": "",
        "street": "",
        "complex": "",
        "landmark": "",
        "untagged": "",
    }

    if not address_str or not address_str.strip():
        return result

    addr = address_str.strip()

    # --- PIN code (6-digit Indian postal code) ---
    # Consume an optional "PIN Code"/"PIN" label too, so it doesn't linger as junk.
    pin_match = re.search(r'(?:\bpin(?:\s*code)?\s*[:.\-]?\s*)?\b(\d{6})\b', addr, re.I)
    if pin_match:
        result["pin"] = pin_match.group(1)
        addr = addr[:pin_match.start()] + addr[pin_match.end():]

    # --- State (consume an optional preceding "State" label) ---
    addr_lower = addr.lower()
    for state in sorted(INDIAN_STATES, key=len, reverse=True):
        pattern = r'(?:\bstate\s*[:.\-]?\s*)?\b' + re.escape(state) + r'\b'
        m = re.search(pattern, addr_lower)
        if m:
            result["state"] = state.title()
            addr = addr[:m.start()] + addr[m.end():]
            break

    # --- Sub-district / tehsil (remove FIRST so the plain "District" match below
    # grabs the real district value, not the sub-district's) ---
    subdist_match = re.search(r'\bsub[\s.\-]*dist(?:rict)?[\s.:~-]*([A-Za-z\s]+)', addr, re.I)
    if subdist_match:
        addr = addr[:subdist_match.start()] + addr[subdist_match.end():]

    # --- District ---
    dist_match = re.search(r'\b(?:dist(?:rict)?|distt?)[\s.:~-]*([A-Za-z\s]+)', addr, re.I)
    if dist_match:
        result["district"] = dist_match.group(1).strip().strip(',').strip()
        addr = addr[:dist_match.start()] + addr[dist_match.end():]

    # Work with comma-separated parts for the remaining fields
    parts = [p.strip() for p in re.split(r'[,\n]+', addr) if p.strip()]
    remaining = []

    for part in parts:
        part_lower = part.lower().strip()

        # --- House number (e.g., "H.No 123", "No. 45", "1-2-3/4", "#123") ---
        if not result["house"] and re.match(
            r'^(?:h\.?\s*no\.?\s*|no\.?\s*|#\s*)?\d[\d\-/\\a-zA-Z]*$', part_lower
        ):
            result["house"] = part.strip()
            continue

        # --- Floor ---
        if not result["floor"] and re.search(
            r'\b(\d+\s*(?:st|nd|rd|th)\s*floor|ground\s*floor|basement)\b', part_lower
        ):
            result["floor"] = part.strip()
            continue

        # --- Landmark ---
        if not result["landmark"] and re.match(
            r'\b(?:near|behind|opp(?:osite)?|beside|adjacent|next\s+to|in\s+front\s+of)\b',
            part_lower
        ):
            result["landmark"] = part.strip()
            continue

        # --- Street ---
        if not result["street"] and re.search(
            r'\b(?:road|rd|street|st|lane|marg|path|gali|gully|chowk|cross)\b', part_lower
        ):
            result["street"] = part.strip()
            continue

        # --- Building ---
        if not result["building"] and re.search(
            r'\b(?:building|bldg|tower|plaza|bhawan|bhavan|mansion|house)\b', part_lower
        ):
            result["building"] = part.strip()
            continue

        # --- Complex (apartment/society/complex) ---
        if not result["complex"] and re.search(
            r'\b(?:apartment|apt|society|complex|residency|enclave|heights|villa|park)\b',
            part_lower
        ):
            result["complex"] = part.strip()
            continue

        # --- Locality (nagar/colony/sector/ward/area/mohalla etc.) ---
        if not result["locality"] and re.search(
            r'\b(?:nagar|colony|sector|ward|area|mohalla|mohala|puram|puri|bagh|'
            r'vihar|kunj|block|phase|extension|extn|layout|scheme|circle|'
            r'town|village|vill|post|po|tehsil|taluka|mandal|hobli)\b',
            part_lower
        ):
            result["locality"] = part.strip()
            continue

        # --- City (capitalized word that doesn't match other patterns) ---
        # Assign first unmatched part with 2+ alpha words or a single capitalized word as city
        if not result["city"] and re.match(r'^[A-Za-z][A-Za-z\s]+$', part.strip()):
            result["city"] = part.strip()
            continue

        remaining.append(part.strip())

    # Drop pure label remnants left over from extraction (e.g. a stray "State",
    # "Sub", "PIN Code") so they don't pollute untagged.
    _label_only = {'state', 'sub', 'dist', 'district', 'sub district', 'pin',
                   'pin code', 'vtc', 'po', 'post', 'tehsil', 'taluka', 'mandal'}
    remaining = [r for r in remaining if r.lower().strip(' :.-') not in _label_only]

    # Anything left goes to untagged
    untagged = ", ".join(remaining).strip(", ").strip()
    result["untagged"] = untagged

    # Clean up trailing/leading commas and whitespace in all fields
    for key in result:
        result[key] = re.sub(r'^[\s,]+|[\s,]+$', '', result[key])

    return result
