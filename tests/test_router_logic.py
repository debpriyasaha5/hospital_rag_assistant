import pytest

from utils.utils import _build_retrieval_query, _choose_route


def test_build_retrieval_query_uses_full_recent_history():
    history = [
        ("what departments are there?", "answer a"),
        ("what are the facilities?", "answer b"),
    ]

    result = _build_retrieval_query("how many departments are present?", history)

    assert "what departments are there?" in result
    assert "what are the facilities?" in result
    assert "how many departments are present?" in result


@pytest.mark.parametrize(
    ("question", "expected_route"),
    [
        ("how many departments and what departments are present", "keyword"),
        ("what departments are present", "keyword"),
        ("what is the consultation fee", "keyword"),
        ("show me all neurology doctors", "keyword"),
        ("tell me about this hospital", "hybrid"),
        ("explain how emergency care works", "vector"),
        ("tell me about the hospital and list all departments", "hybrid"),
        ("hi, what are the visiting hours?", "keyword"),
    ],
)
def test_choose_route(question, expected_route):
    assert _choose_route(question) == expected_route
