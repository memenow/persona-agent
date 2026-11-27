"""REST routes for persona CRUD and file upload."""

import json
import logging
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import ValidationError

from persona_agent.api.dependencies import get_persona_repository
from persona_agent.api.schemas import (
    CreatePersonaRequest,
    PersonaDetailResponse,
    PersonaListResponse,
    PersonaResponse,
    SuccessResponse,
    UpdatePersonaRequest,
)
from persona_agent.personas.repository import PersonaPersistenceError, PersonaRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/personas", tags=["personas"])

PersonaRepoDep = Annotated[PersonaRepository, Depends(get_persona_repository)]
UploadFileDep = Annotated[UploadFile, File(...)]


@router.get("", response_model=PersonaListResponse)
async def list_personas(persona_repo: PersonaRepoDep) -> PersonaListResponse:
    return PersonaListResponse(
        personas=[
            PersonaResponse(
                id=persona.id,
                name=persona.name,
                description=persona.description,
            )
            for persona in persona_repo.list_personas()
        ]
    )


@router.get("/{persona_id}", response_model=PersonaDetailResponse)
async def get_persona(
    persona_id: str, persona_repo: PersonaRepoDep
) -> PersonaDetailResponse:
    persona = persona_repo.get_persona(persona_id)
    if persona is None:
        raise HTTPException(
            status_code=404, detail=f"Persona with ID {persona_id} not found"
        )
    return PersonaDetailResponse(
        id=persona.id,
        name=persona.name,
        description=persona.description,
        personal_background=persona.personal_background,
        language_style=persona.language_style,
        knowledge_domains=persona.knowledge_domains,
        interaction_samples=persona.interaction_samples,
    )


@router.post("", response_model=PersonaResponse, status_code=201)
async def create_persona(
    request: CreatePersonaRequest, persona_repo: PersonaRepoDep
) -> PersonaResponse:
    new_persona = persona_repo.add_persona(request.model_dump(exclude_unset=True))

    try:
        persona_repo.save_persona(new_persona, fmt="json")
    except Exception as exc:
        persona_repo.personas.pop(new_persona.id, None)
        logger.exception("Error saving persona %s", new_persona.id)
        raise HTTPException(status_code=500, detail="Error saving persona") from exc

    return PersonaResponse(
        id=new_persona.id,
        name=new_persona.name,
        description=new_persona.description,
    )


@router.put("/{persona_id}", response_model=PersonaResponse)
async def update_persona(
    persona_id: str,
    request: UpdatePersonaRequest,
    persona_repo: PersonaRepoDep,
) -> PersonaResponse:
    previous = persona_repo.get_persona(persona_id)
    if previous is None:
        raise HTTPException(
            status_code=404, detail=f"Persona with ID {persona_id} not found"
        )

    update_data = request.model_dump(exclude_unset=True)
    updated = persona_repo.update_persona(persona_id, update_data)
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to update persona")

    try:
        persona_repo.save_persona(updated, fmt="json")
    except Exception as exc:
        persona_repo.personas[persona_id] = previous
        logger.exception("Error saving persona %s", updated.id)
        raise HTTPException(status_code=500, detail="Error saving persona") from exc

    return PersonaResponse(
        id=updated.id,
        name=updated.name,
        description=updated.description,
    )


@router.delete("/{persona_id}", response_model=SuccessResponse)
async def delete_persona(
    persona_id: str, persona_repo: PersonaRepoDep
) -> SuccessResponse:
    try:
        deleted = persona_repo.delete_persona(persona_id)
    except PersonaPersistenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Persona with ID {persona_id} not found"
        )

    return SuccessResponse(message=f"Persona with ID {persona_id} deleted successfully")


@router.post("/upload", response_model=PersonaResponse, status_code=201)
async def upload_persona(
    file: UploadFileDep, persona_repo: PersonaRepoDep
) -> PersonaResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided")

    if not file.filename.endswith((".json", ".yaml", ".yml")):
        raise HTTPException(status_code=400, detail="File must be a JSON or YAML file")

    try:
        content_str = (await file.read()).decode("utf-8")
        persona_data = (
            json.loads(content_str)
            if file.filename.endswith(".json")
            else yaml.safe_load(content_str)
        )
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="File must be UTF-8 encoded"
        ) from None
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file") from None
    except yaml.YAMLError:
        raise HTTPException(status_code=400, detail="Invalid YAML file") from None

    if not isinstance(persona_data, dict):
        raise HTTPException(
            status_code=422, detail="Persona file must contain an object"
        )

    try:
        new_persona = persona_repo.add_persona(persona_data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from None

    try:
        persona_repo.save_persona(new_persona, fmt="json")
    except Exception as exc:
        persona_repo.personas.pop(new_persona.id, None)
        logger.exception("Error saving uploaded persona %s", new_persona.id)
        raise HTTPException(status_code=500, detail="Error saving persona") from exc

    return PersonaResponse(
        id=new_persona.id,
        name=new_persona.name,
        description=new_persona.description,
    )
