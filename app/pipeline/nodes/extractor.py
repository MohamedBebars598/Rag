from __future__ import annotations

import json
import io

from pypdf import PdfReader

from app.pipeline.schemas import CVData, PipelineState
from app.services.openrouter import chat_completion

EXTRACTION_SYSTEM_PROMPT = """You are an expert HR data extractor. 
Extract all information from the CV text provided and return it as valid JSON.
Translate all content to English if it is in another language.
Return ONLY the JSON object with no additional text or markdown."""

EXTRACTION_USER_TEMPLATE = """Extract all information from this CV and return a JSON object with exactly these fields:
{{
  "full_name": "candidate full name",
  "email": "email address or empty string",
  "phone": "phone number or empty string",
  "location": "city/country or empty string",
  "languages": ["list", "of", "languages"],
  "skills": ["list", "of", "technical", "and", "soft", "skills"],
  "experience": [
    {{
      "title": "job title",
      "company": "company name",
      "start_date": "YYYY-MM or empty",
      "end_date": "YYYY-MM or 'Present' or empty",
      "description": "role description"
    }}
  ],
  "education": [
    {{
      "degree": "degree name",
      "institution": "institution name",
      "field_of_study": "field or empty",
      "graduation_year": "YYYY or empty"
    }}
  ],
  "meta_summary": "2-3 sentence professional summary of this candidate highlighting their key strengths and experience level"
}}

CV Text:
{cv_text}"""


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pypdf."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


async def extract_node(state: PipelineState) -> dict:
    """Node 1: Extract structured CV data from raw PDF bytes via GPT-4.1-mini."""
    pdf_bytes = state["raw_pdf_bytes"]
    filename = state["filename"]
    candidate_id = state["candidate_id"]

    cv_text = _extract_text_from_pdf(pdf_bytes)

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": EXTRACTION_USER_TEMPLATE.format(cv_text=cv_text),
        },
    ]

    raw_json = await chat_completion(
        messages=messages,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        data = {}

    data["candidate_id"] = candidate_id
    data["original_filename"] = filename

    cv_data = CVData(**data).model_dump()

    return {"cv_data": cv_data}
