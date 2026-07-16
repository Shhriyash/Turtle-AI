from apps.turtle_server import _parse_confirmation_answer, _wants_preview


def test_confirmation_answers_are_bare_only():
    assert _parse_confirmation_answer("yes") is True
    assert _parse_confirmation_answer("yes.") is True
    assert _parse_confirmation_answer("no thanks") is False
    assert _parse_confirmation_answer("ok") is True
    assert _parse_confirmation_answer("yes send it") is None
    assert _parse_confirmation_answer("ok, now search for flights to goa") is None
    assert _parse_confirmation_answer("go ahead") is True
    assert _parse_confirmation_answer("go ahead and email bob") is None


def test_preview_answers_are_exact_only():
    assert _wants_preview("show me") is True
    assert _wants_preview("preview") is True
    assert _wants_preview("ok show me the news") is False
    assert _wants_preview("give me more details about the plan") is False
