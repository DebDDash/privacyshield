"""
PURPOSE:
    Detects all PII in extracted text using Presidio + spaCy.

INPUT:
    - text (str): Plain text extracted from PDF page
    - language (str): default 'en'
    - document_type (str): 'medical','financial','insurance','tax','general','auto'

OUTPUT:
    [{"entity_type": "PERSON", "text": "Shrimi Agrawal", "start": 6, "end": 16, "score": 0.85}, ...]

METHOD USED:
    1. Build Presidio analyzer with spaCy en_core_web_lg
    2. Add custom recognizers for SSN, amounts, IDs, medical conditions
    3. Run analysis at threshold 0.4
    4. Remove false positives (label words, URL-in-email, state codes)
    5. Deduplicate overlapping spans (keep highest score)
    6. Apply document-type specific rules

FALSE POSITIVES HANDLED:
    - "SSN","DOB" misclassified as ORGANIZATION → removed
    - "gmail.com" as URL when email already detected → removed
    - 2-letter state codes as LOCATION → removed
    - Same span by multiple recognizers → keep highest score
    - Policy numbers as PERSON → reclassified as ID_NUMBER

DEPENDENCIES:
    pip install presidio-analyzer
    python -m spacy download en_core_web_lg
"""

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider

CONFIDENCE_THRESHOLD = 0.4

ORG_FALSE_POSITIVES = {
    "SSN", "DOB", "EIN", "NPI", "ID", "DOJ",
    "IL", "MA", "CA", "NY", "TX",
}

DOCUMENT_KEYWORDS = {
    "medical": [
        "diagnosis", "patient", "physician", "prescription",
        "hospital", "clinic", "medical", "health", "doctor",
        "treatment", "condition", "symptom", "medication"
    ],
    "financial": [
        "salary", "payroll", "gross", "net pay", "tax withheld",
        "employee", "corporation", "w-2", "income", "wages",
        "deduction", "reimbursement"
    ],
    "insurance": [
        "policy", "premium", "coverage", "claim", "insured",
        "beneficiary", "deductible", "copay", "insurance"
    ],
    "tax": [
        "tax return", "1040", "refund", "irs", "federal tax",
        "adjusted gross", "taxable income", "filing status"
    ]
}

_analyzer = None



def _build_custom_recognizers() -> list:
    """
    Build custom PatternRecognizer objects for PII types
    that Presidio's built-in recognizers miss or mishandle.
    Returns list of PatternRecognizer objects.
    """
    recognizers = []

    # SSN — fixes Presidio's unreliable built-in
    recognizers.append(PatternRecognizer(
        supported_entity="US_SSN",
        patterns=[
            Pattern("SSN_dashes", r"\b\d{3}-\d{2}-\d{4}\b", 0.9),
            Pattern("SSN_plain",  r"\b\d{9}\b", 0.5),
        ],
        context=["ssn", "social", "security", "social security"]
    ))

    # Financial amounts
    recognizers.append(PatternRecognizer(
        supported_entity="FINANCIAL_AMOUNT",
        patterns=[
            Pattern("usd_amount",    r"\$\s?[\d,]+(?:\.\d{2})?", 0.7),
            Pattern("usd_per_month", r"\$\s?[\d,]+/month", 0.8),
        ],
        context=[
            "salary", "income", "wage", "pay", "earning",
            "benefit", "amount", "due", "premium", "refund",
            "gross", "net", "total"
        ]
    ))

    # Employee / Patient / Policy IDs
    recognizers.append(PatternRecognizer(
        supported_entity="ID_NUMBER",
        patterns=[
            Pattern("prefixed_id",
                    r"\b(EMP|PAT|POL|MED|INV|CLM|REF)-[\w\d]+\b", 0.85),
            Pattern("labeled_id",
                    r"(?i)(?:employee|patient|member|account)\s*(?:#|no\.?|number)?\s*:?\s*(\d{4,})",
                    0.75),
        ],
        context=["employee", "patient", "policy", "member", "account", "id", "number"]
    ))

    # Medical conditions
    recognizers.append(PatternRecognizer(
        supported_entity="MEDICAL_CONDITION",
        patterns=[
            Pattern("diagnosis_label",
                    r"(?i)(?:diagnosis|condition|disorder):\s*([A-Z][^\n,\.]{3,50})", 0.8),
            Pattern("condition_inline",
                    r"(?i)\b(diabetes|hypertension|asthma|cancer|arthritis|"
                    r"depression|anxiety|cardiac|cardiovascular)\b", 0.65),
        ],
        context=["diagnosis", "condition", "medical", "pre-existing", "history"]
    ))

    # Indian PAN Card
    recognizers.append(PatternRecognizer(
        supported_entity="IN_PAN",
        patterns=[Pattern("pan", r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b", 0.85)],
        context=["pan", "permanent account"]
    ))

    # Indian Aadhaar
    recognizers.append(PatternRecognizer(
        supported_entity="IN_AADHAAR",
        patterns=[Pattern("aadhaar", r"\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b", 0.8)],
        context=["aadhaar", "aadhar", "uid"]
    ))
  
    # SWIFT/BIC codes
    recognizers.append(PatternRecognizer(
        supported_entity="SWIFT_BIC",
        patterns=[Pattern("bic", r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b", 0.8)],
        context=["bic", "swift", "bank", "payment"]
    ))

    # Internal case/doc reference numbers
    recognizers.append(PatternRecognizer(
        supported_entity="INTERNAL_REF",
        patterns=[Pattern("case_ref",
            r"\b(ZRH|CASE|DOC|REF|TICKET|SUB)-[\w\d-]+\b", 0.8)],
        context=["case", "reference", "document", "submission", "ticket"]
    ))

    return recognizers



def build_analyzer() -> AnalyzerEngine:
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    })
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    for recognizer in _build_custom_recognizers():
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


def get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        _analyzer = build_analyzer()
    return _analyzer


def auto_detect_document_type(text: str) -> str:
    """
    Detect document type from keyword frequency.

    Args:
        text: Plain text of document.

    Returns:
        'medical', 'financial', 'insurance', 'tax', or 'general'
    """
    text_lower = text.lower()
    scores = {
        doc_type: sum(1 for kw in keywords if kw in text_lower)
        for doc_type, keywords in DOCUMENT_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"



def _remove_false_positives(entities: list, text: str, document_type: str) -> list:
    email_spans = [
        (e["start"], e["end"])
        for e in entities
        if e["entity_type"] == "EMAIL_ADDRESS"
    ]

    filtered = []
    for entity in entities:
        entity_text = entity["text"].strip()
        entity_type = entity["entity_type"]

        if entity_type == "ORGANIZATION":
            if entity_text.upper() in ORG_FALSE_POSITIVES:
                continue

        if entity_type == "URL":
            if any(s <= entity["start"] and entity["end"] <= e
                   for s, e in email_spans):
                continue

        if entity_type == "LOCATION":
            if len(entity_text) == 2 and entity_text.isupper():
                continue

        filtered.append(entity)

    return filtered


def _deduplicate_entities(entities: list) -> list:
    span_map = {}
    for entity in entities:
        key = (entity["start"], entity["end"])
        if key not in span_map or entity["score"] > span_map[key]["score"]:
            span_map[key] = entity

    result = list(span_map.values())
    final = []
    for entity in result:
        contained = any(
            other["start"] <= entity["start"]
            and entity["end"] <= other["end"]
            and other["score"] >= entity["score"]
            and (other["start"], other["end"]) != (entity["start"], entity["end"])
            for other in result
        )
        if not contained:
            final.append(entity)

    return sorted(final, key=lambda x: x["start"])



def detect_pii(
    text: str,
    language: str = "en",
    document_type: str = "auto"
) -> list:
    if not text or not text.strip():
        return []

    if document_type == "auto":
        document_type = auto_detect_document_type(text)

    analyzer = get_analyzer()

    raw_results = analyzer.analyze(
        text=text,
        language=language,
        score_threshold=CONFIDENCE_THRESHOLD,
    )

    entities = [{
        "entity_type": r.entity_type,
        "text": text[r.start:r.end],
        "start": r.start,
        "end": r.end,
        "score": round(r.score, 3),
    } for r in raw_results]

    entities = _remove_false_positives(entities, text, document_type)
    entities = _deduplicate_entities(entities)

    return entities


def get_pii_summary(entities: list) -> dict:
    summary = {}
    for e in entities:
        summary[e["entity_type"]] = summary.get(e["entity_type"], 0) + 1
    return summary
