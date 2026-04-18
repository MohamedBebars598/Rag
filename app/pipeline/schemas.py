from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── CV extraction schema (output of Node 1) ──────────────────────────────────

class ExperienceEntry(BaseModel):
    title: str = ""
    company: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""


class EducationEntry(BaseModel):
    degree: str = ""
    institution: str = ""
    field_of_study: str = ""
    graduation_year: str = ""


class CVData(BaseModel):
    candidate_id: str = ""
    full_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    languages: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    meta_summary: str = ""
    original_filename: str = ""


# ── Chunk schema (output of Node 2) ──────────────────────────────────────────

class Chunk(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── LangGraph pipeline state ──────────────────────────────────────────────────

class PipelineState(TypedDict):
    raw_pdf_bytes: bytes
    filename: str
    candidate_id: str
    cv_data: dict[str, Any]
    chunks: list[dict[str, Any]]
    stored_ids: list[str]
