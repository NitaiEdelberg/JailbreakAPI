"""Input validation on the request model."""
import pytest
from pydantic import ValidationError

from models.message_model import Message, MAX_LEN


def test_accepts_normal_text():
    assert Message(text="hello there").text == "hello there"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_rejects_blank(bad):
    with pytest.raises(ValidationError):
        Message(text=bad)


def test_rejects_too_long():
    with pytest.raises(ValidationError):
        Message(text="a" * (MAX_LEN + 1))


def test_accepts_at_limit():
    assert Message(text="a" * MAX_LEN).text.startswith("a")
