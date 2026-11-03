import re


HISTORY_LIMIT = 6
session_histories = {}

STRUCTURED_ENTITIES = (
    "department", "doctor", "bed", "ward", "test", "package", "ambulance",
    "service", "facility", "insurance", "specialist", "consultant", "icu",
    "opd", "lab", "pharmacy", "medicine", "emergency",
)

EXACT_FACT_TERMS = (
    "fee", "fees", "price", "cost", "charge", "charges", "contact", "phone",
    "extension", "room", "address", "location", "timing", "schedule", "hours",
    "qualification", "experience", "availability", "opening", "visiting",
)

OVERVIEW_TERMS = (
    "tell me about this hospital", "about this hospital", "hospital overview",
    "summary of the hospital", "describe the hospital", "what kind of hospital",
    "hospital details", "hospital information", "overall overview",
)

SEMANTIC_TERMS = (
    "treatment", "procedure", "policy", "policies", "care", "clinical",
    "infrastructure", "support", "process", "how does", "explain",
)

GENERAL_PHRASES = {
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "thanks", "thank you", "bye", "goodbye", "who are you",
}


def _normalize_question(question):
    return re.sub(r"\s+", " ", str(question or "").strip().lower())


def _contains_any(text, terms):
    return any(term in text for term in terms)


def _is_general_question(question):
    return _normalize_question(question).strip(" .,!?;:") in GENERAL_PHRASES


def _is_overview_question(question):
    question = _normalize_question(question)
    return _contains_any(question, OVERVIEW_TERMS) or (
        "hospital" in question
        and _contains_any(question, ("about", "overview", "details", "summary"))
    )


def _is_structured_fact_question(question):
    question = _normalize_question(question)
    if not question:
        return False

    entity_pattern = "|".join(re.escape(entity) for entity in STRUCTURED_ENTITIES)
    exact_pattern = "|".join(re.escape(term) for term in EXACT_FACT_TERMS)
    patterns = (
        rf"\b(how many|number of|count of)\b.*\b({entity_pattern})s?\b",
        rf"\b(list|show me|which|who are|what are)\b.*\b({entity_pattern})s?\b",
        rf"\bwhat\b.*\b({entity_pattern})s?\b.*\b(are present|are available|do you have|does the hospital have|do you provide)\b",
        rf"\b(what is|what are|how much|where is|when is|when are)\b.*\b({exact_pattern})\b",
        rf"\b({exact_pattern})\b",
    )
    return any(re.search(pattern, question) for pattern in patterns)


def _choose_route(question):
    question = _normalize_question(question)
    if _is_general_question(question):
        return "general"

    has_overview = _is_overview_question(question)
    has_structured_fact = _is_structured_fact_question(question)
    if has_overview and has_structured_fact:
        return "hybrid"
    if has_structured_fact:
        return "keyword"
    if has_overview:
        return "hybrid"
    if _contains_any(question, SEMANTIC_TERMS):
        return "vector"
    return "hybrid"


def _get_history(session_id):
    if not session_id:
        return []
    return session_histories.setdefault(session_id, [])


def _remember(session_id, question, answer):
    if not session_id:
        return
    history = _get_history(session_id)
    history.append((question, answer))
    del history[:-HISTORY_LIMIT]


def clear_chat_history(session_id):
    session_histories.pop(session_id, None)


def _build_retrieval_query(question, chat_history):
    if not chat_history:
        return question

    recent_questions = [
        previous_question
        for previous_question, _ in chat_history[-HISTORY_LIMIT:]
        if previous_question and previous_question.strip()
    ]
    if not recent_questions:
        return question
    return f"{question} (related context: {' | '.join(recent_questions)})"
import re
import nltk
from nltk.corpus import nps_chat

# Download corpus (only required once)
nltk.download('nps_chat')

# Extract general conversational phrases
GENERAL_CLASSES = {"Greet", "Bye", "Emotion"}

general_phrases = set()

for post in nps_chat.xml_posts():
    if post.get("class") in GENERAL_CLASSES:
        phrase = post.text.lower().strip()

        if phrase:
            general_phrases.add(phrase)


def _is_general_question(question):
    q = question.lower().strip()

    if q in general_phrases:
        return True

    # Handle messages containing a greeting followed by additional text
    for phrase in general_phrases:
        if q.startswith(phrase + " "):
            return True

    return False

def _is_exact_structured_fact_question(question):
    q = question.lower()

    exact_fact_terms = [
        "price", "fee", "cost", "doctor", "doctor name", "doctor id", "department",
        "contact", "extension", "room", "location", "timing", "schedule", "days",
        "qualification", "experience", "bed", "icu", "ambulance", "insurance",
        "lab", "test", "package", "pharmacy", "medicine", "opd", "ward",
        "hospital address", "working hours", "parking", "emergency"
    ]

    if _contains_any(q, exact_fact_terms):
        return True

    exact_patterns = [
        r"\bwhat is\b.*\b(price|fee|cost|contact|extension|room|location|timing|days)\b",
        r"\bhow much\b.*\b(price|fee|cost)\b",
        r"\bhow many\b.*\b(departments|beds|doctors|tests|packages|ambulances)\b",
        r"\bwhich\b.*\b(doctor|department|test|package|insurance)\b",
        r"\bwho is\b.*\b(doctor|consultant|specialist)\b",
        r"\bshow me\b.*\b(doctors|departments|tests|packages|ambulances)\b",
        r"\blist\b.*\b(doctors|departments|tests|packages|ambulances)\b",
    ]

    if any(re.search(pattern, q) for pattern in exact_patterns):
        return True

    return False

def _is_broad_overview_question(question):
    q = question.lower()
    broad_terms = [
        "tell me about this hospital",
        "about this hospital",
        "hospital overview",
        "summary of the hospital",
        "what kind of hospital is this",
        "describe the hospital",
        "what departments are present",
        "how many departments",
        "what services are available"
    ]
    return _contains_any(q, broad_terms) or (
        "hospital" in q and ("overview" in q or "details" in q or "about" in q)
    )

def _contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)