from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.pipeline.graph import pipeline
from app.services.qdrant_client import ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_collection()
    yield


app = FastAPI(
    title="HR RAG - CV Ingestion Pipeline",
    description=(
        "Upload a PDF CV to extract, chunk, embed, and store candidate data "
        "in Qdrant for semantic talent search."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/ingest", summary="Ingest a candidate CV (PDF)")
async def ingest(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Received: {file.content_type}",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    candidate_id = str(uuid.uuid4())

    initial_state = {
        "raw_pdf_bytes": pdf_bytes,
        "filename": file.filename or "unknown.pdf",
        "candidate_id": candidate_id,
        "cv_data": {},
        "chunks": [],
        "stored_ids": [],
    }

    try:
        result = await pipeline.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {exc}",
        ) from exc

    cv_data = result.get("cv_data", {})

    return JSONResponse(
        status_code=200,
        content={
            "candidate_id": candidate_id,
            "full_name": cv_data.get("full_name", ""),
            "meta_summary": cv_data.get("meta_summary", ""),
            "chunks_created": len(result.get("chunks", [])),
            "stored_ids": result.get("stored_ids", []),
        },
    )


@app.get("/health", summary="Health check")
async def health() -> dict:
    return {"status": "ok"}
