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
