from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Guardrails on the input itself: reject empty prompts and cap the size so a
# giant payload can't be used to hammer the scanners.
MAX_LEN = 10_000
MAX_BATCH_ITEMS = 25
MAX_BATCH_CHARS = 100_000

# Where the text came from. "user" is what the person talking to the model typed.
# Everything else is content the model will READ (a retrieved RAG chunk, a web
# page an agent opened, a tool result, an email, a pasted document). Untrusted
# content gets the indirect-injection rules, which would be noise on user input.
Source = Literal["user", "document", "web", "email", "tool", "rag"]


class Message(BaseModel):
    text: str
    source: Source = "user"

    @field_validator("text")
    @classmethod
    def not_blank_and_bounded(cls, v: str) -> str:
        if v is None or not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > MAX_LEN:
            raise ValueError(f"text must be at most {MAX_LEN} characters")
        return v


class BatchItem(Message):
    id: Optional[str] = Field(default=None, max_length=200)


class BatchRequest(BaseModel):
    items: List[BatchItem] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)

    @model_validator(mode="after")
    def bounded_total(self):
        total = sum(len(i.text) for i in self.items)
        if total > MAX_BATCH_CHARS:
            raise ValueError(f"batch text must total at most {MAX_BATCH_CHARS} characters")
        return self
