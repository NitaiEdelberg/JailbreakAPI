from pydantic import BaseModel, field_validator

# Guardrails on the input itself: reject empty prompts and cap the size so a
# giant payload can't be used to hammer the scanners.
MAX_LEN = 10_000


class Message(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def not_blank_and_bounded(cls, v: str) -> str:
        if v is None or not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > MAX_LEN:
            raise ValueError(f"text must be at most {MAX_LEN} characters")
        return v
