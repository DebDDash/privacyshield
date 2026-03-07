from collections import defaultdict

TOKEN_PREFIX_MAP = {
    "PERSON":            "NAME",
    "LOCATION":          "ADDRESS",
    "EMAIL_ADDRESS":     "EMAIL",
    "PHONE_NUMBER":      "PHONE",
    "US_SSN":            "SSN",
    "CREDIT_CARD":       "CARD",
    "IBAN_CODE":         "IBAN",
    "DATE_TIME":         "DATE",
    "US_BANK_NUMBER":    "BANK",
    "US_DRIVER_LICENSE": "DL",
    "US_ITIN":           "ITIN",
    "US_PASSPORT":       "PASSPORT",
    "MEDICAL_CONDITION": "MEDICAL",
    "FINANCIAL_AMOUNT":  "AMOUNT",
    "ID_NUMBER":         "ID",
    "CH_AHV":            "AHV",
    "SWIFT_BIC":         "BIC",
    "INTERNAL_REF":      "REF",
    "IN_PAN":            "PAN",
    "ORGANIZATION":      "ORG",
    "NRP":               "NRP",
    "URL":               "URL",
}

LABEL_STRIP_ENTITIES = {"MEDICAL_CONDITION"}
LABEL_PREFIXES = [
    "diagnosis:", "diagnose:", "condition:", "disorder:",
    "erkrankung:", "diagnosi:"
]

def _get_token_prefix(entity_type):
    return TOKEN_PREFIX_MAP.get(entity_type, entity_type)

def _strip_label(text, entity_type):
    if entity_type not in LABEL_STRIP_ENTITIES:
        return text
    text_lower = text.lower()
    for prefix in LABEL_PREFIXES:
        if text_lower.startswith(prefix):
            return text[len(prefix):].strip()
    return text

def _seed_counters_from_map(token_map):
    counters = defaultdict(int)
    for token_id in token_map.keys():
        parts = token_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            prefix = parts[0]
            counters[prefix] = max(counters[prefix], int(parts[1]))
    return counters

def redact_text(text, pii_entities, existing_token_map=None):
    if not pii_entities:
        return text, existing_token_map or {}

    token_map = dict(existing_token_map) if existing_token_map else {}
    value_to_token = {v: k for k, v in token_map.items()}
    type_counters = _seed_counters_from_map(token_map)

    sorted_entities = sorted(pii_entities, key=lambda x: x["start"], reverse=True)
    text_chars = list(text)

    for entity in sorted_entities:
        original_value = entity["text"]
        entity_type = entity["entity_type"]
        start = entity["start"]
        end = entity["end"]

        stored_value = _strip_label(original_value, entity_type)
        prefix = _get_token_prefix(entity_type)

        if stored_value in value_to_token:
            token_id = value_to_token[stored_value]
        elif original_value in value_to_token:
            token_id = value_to_token[original_value]
        else:
            type_counters[prefix] += 1
            token_id = f"{prefix}_{type_counters[prefix]}"
            token_map[token_id] = stored_value
            value_to_token[stored_value] = token_id

        replacement = list(f"[{token_id}]")
        text_chars[start:end] = replacement

    redacted_text = "".join(text_chars)
    return redacted_text, token_map

def restore_text(redacted_text, token_map):
    restored = redacted_text
    sorted_tokens = sorted(token_map.keys(), key=len, reverse=True)
    for token_id in sorted_tokens:
        placeholder = f"[{token_id}]"
        restored = restored.replace(placeholder, token_map[token_id])
    return restored

def get_redaction_stats(token_map):
    counts = defaultdict(int)
    for token_id in token_map.keys():
        parts = token_id.rsplit("_", 1)
        if len(parts) == 2:
            counts[parts[0]] += 1
    counts["TOTAL"] = len(token_map)
    return dict(counts)
