from __future__ import annotations

from app.pipeline.schemas import Chunk, PipelineState


def _base_meta(cv_data: dict, section: str) -> dict:
    return {
        "candidate_id": cv_data.get("candidate_id", ""),
        "original_filename": cv_data.get("original_filename", ""),
        "full_name": cv_data.get("full_name", ""),
        "section": section,
    }


def _profile_chunk(cv_data: dict) -> Chunk:
    lines = [
        f"Name: {cv_data.get('full_name', '')}",
        f"Email: {cv_data.get('email', '')}",
        f"Phone: {cv_data.get('phone', '')}",
        f"Location: {cv_data.get('location', '')}",
        f"Languages: {', '.join(cv_data.get('languages', []))}",
    ]
    summary = cv_data.get("meta_summary", "")
    if summary:
        lines.append(f"Summary: {summary}")
    return Chunk(
        text="\n".join(line for line in lines if line.split(": ", 1)[-1]),
        metadata=_base_meta(cv_data, "profile"),
    )


def _skills_chunk(cv_data: dict) -> Chunk | None:
    skills = cv_data.get("skills", [])
    if not skills:
        return None
    return Chunk(
        text=f"Skills: {', '.join(skills)}",
        metadata=_base_meta(cv_data, "skills"),
    )


def _experience_chunks(cv_data: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, exp in enumerate(cv_data.get("experience", [])):
        lines = [
            f"Job Title: {exp.get('title', '')}",
            f"Company: {exp.get('company', '')}",
            f"Period: {exp.get('start_date', '')} - {exp.get('end_date', '')}",
            f"Description: {exp.get('description', '')}",
        ]
        meta = _base_meta(cv_data, "experience")
        meta["experience_index"] = i
        meta["company"] = exp.get("company", "")
        meta["job_title"] = exp.get("title", "")
        chunks.append(
            Chunk(
                text="\n".join(line for line in lines if line.split(": ", 1)[-1]),
                metadata=meta,
            )
        )
    return chunks


def _education_chunks(cv_data: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, edu in enumerate(cv_data.get("education", [])):
        lines = [
            f"Degree: {edu.get('degree', '')}",
            f"Institution: {edu.get('institution', '')}",
            f"Field of Study: {edu.get('field_of_study', '')}",
            f"Graduation Year: {edu.get('graduation_year', '')}",
        ]
        meta = _base_meta(cv_data, "education")
        meta["education_index"] = i
        meta["institution"] = edu.get("institution", "")
        chunks.append(
            Chunk(
                text="\n".join(line for line in lines if line.split(": ", 1)[-1]),
                metadata=meta,
            )
        )
    return chunks


async def chunk_node(state: PipelineState) -> dict:
    """Node 2: Split CVData into field-aware semantic chunks."""
    cv_data = state["cv_data"]

    all_chunks: list[Chunk] = []

    all_chunks.append(_profile_chunk(cv_data))

    skills = _skills_chunk(cv_data)
    if skills:
        all_chunks.append(skills)

    all_chunks.extend(_experience_chunks(cv_data))
    all_chunks.extend(_education_chunks(cv_data))

    return {"chunks": [c.model_dump() for c in all_chunks]}
