"""Data schemas and validation contracts for Multilingual Topic & Headline Generation."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Language(str, Enum):
    """Supported African languages for the competition."""

    HAUSA = "hau"
    IGBO = "ibo"
    PIDGIN = "pcm"
    YORUBA = "yor"

    @property
    def full_name(self) -> str:
        """Return the natural full name of the language to enrich LLM prompts."""
        mapping = {
            Language.HAUSA: "Hausa",
            Language.IGBO: "Igbo",
            Language.PIDGIN: "Nigerian Pidgin",
            Language.YORUBA: "Yoruba",
        }
        return mapping[self]


class TopicLabel(str, Enum):
    """The 7 mutually exclusive target categories for Task A."""

    BUSINESS = "business"
    HEALTH = "health"
    POLITICS = "politics"
    RELIGION = "religion"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    TECHNOLOGY = "technology"

    @classmethod
    def from_str(cls, value: str) -> "TopicLabel":
        """Normalize string to enum or raise ValueError."""
        cleaned = value.strip().lower()
        for member in cls:
            if member.value == cleaned:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Invalid topic label '{value}'. Permitted: {valid}")


class NewsArticle(BaseModel):
    """Audit-ready validated news record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1, description="Unique row identifier")
    lang: Language = Field(..., description="Language ISO code")
    text: str = Field(..., min_length=1, description="News body text")
    headline: Optional[str] = Field(
        default=None, description="News headline (Task B ground truth)"
    )
    category: Optional[TopicLabel] = Field(
        default=None, description="Topic label (Task A ground truth)"
    )
    url: Optional[str] = Field(
        default=None, description="Source article URL if available"
    )
    split: Optional[str] = Field(default=None, description="Dataset split partition")

    @field_validator("id", "text", mode="before")
    @classmethod
    def validate_non_empty_str(cls, v: object) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field must be a non-empty string.")
        return v.strip()

    @field_validator("headline", mode="before")
    @classmethod
    def normalize_headline(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Headline must be a string if provided.")
        stripped = v.strip()
        return stripped if stripped else None

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: object) -> Optional[TopicLabel]:
        if v is None or v == "":
            return None
        if isinstance(v, TopicLabel):
            return v
        if isinstance(v, str):
            return TopicLabel.from_str(v)
        raise ValueError(f"Unsupported category type: {type(v)}")

    @field_validator("lang", mode="before")
    @classmethod
    def normalize_lang(cls, v: object) -> Language:
        if isinstance(v, Language):
            return v
        if isinstance(v, str):
            cleaned = v.strip().lower()
            try:
                return Language(cleaned)
            except ValueError as err:
                valid_codes = [lang_item.value for lang_item in Language]
                raise ValueError(
                    f"Unsupported language code '{v}'. Permitted: {valid_codes}"
                ) from err
        raise ValueError(f"Unsupported language type: {type(v)}")
