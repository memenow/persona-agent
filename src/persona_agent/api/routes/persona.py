"""API routes for persona management."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import json
import yaml

from src.persona_agent.api.models import (
    PersonaResponse,
    PersonaListResponse,
    PersonaDetailResponse,
    CreatePersonaRequest,
    UpdatePersonaRequest,
    SuccessResponse,
    ErrorResponse
)
from src.persona_agent.api.dependencies import get_persona_manager


router = APIRouter(prefix="/personas", tags=["personas"])


@router.get(
    "",
    response_model=PersonaListResponse,
    summary="List all personas",
    description="Returns a list of all available personas."
)
async def list_personas(
    persona_manager=Depends(get_persona_manager)
):
    """List all available personas."""
    personas = persona_manager.list_personas()
    return PersonaListResponse(personas=[
        PersonaResponse(
            id=persona["id"],
            name=persona["name"],
            description=persona["description"]
        )
        for persona in personas
    ])


@router.get(
    "/{persona_id}",
    response_model=PersonaDetailResponse,
    summary="Get persona details",
    description="Returns detailed information about a specific persona."
)
async def get_persona(
    persona_id: str,
    persona_manager=Depends(get_persona_manager)
):
    """Get detailed information about a specific persona."""
    persona = persona_manager.get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona with ID {persona_id} not found")
    
    return PersonaDetailResponse(
        id=persona.id,
        name=persona.name,
        description=persona.description,
        personal_background=persona.personal_background,
        language_style=persona.language_style,
        knowledge_domains=persona.knowledge_domains,
        interaction_samples=persona.interaction_samples
    )


@router.post(
    "",
    response_model=PersonaResponse,
    status_code=201,
    summary="Create a new persona",
    description="Creates a new persona with the provided information."
)
async def create_persona(
    persona: CreatePersonaRequest,
    persona_manager=Depends(get_persona_manager)
):
    """Create a new persona."""
    persona_data = persona.dict(exclude_unset=True)
    new_persona = persona_manager.add_persona(persona_data)
    
    # Save the persona to a file
    try:
        persona_manager.save_persona(new_persona, format='json')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving persona: {e}")
    
    return PersonaResponse(
        id=new_persona.id,
        name=new_persona.name,
        description=new_persona.description
    )


@router.put(
    "/{persona_id}",
    response_model=PersonaResponse,
    summary="Update a persona",
    description="Updates an existing persona with the provided information."
)
async def update_persona(
    persona_id: str,
    persona: UpdatePersonaRequest,
    persona_manager=Depends(get_persona_manager)
):
    """Update an existing persona."""
    existing_persona = persona_manager.get_persona(persona_id)
    if not existing_persona:
        raise HTTPException(status_code=404, detail=f"Persona with ID {persona_id} not found")
    
    # Update only provided fields
    update_data = persona.dict(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:
            setattr(existing_persona, field, value)
    
    updated_persona = persona_manager.update_persona(persona_id, existing_persona.dict())
    if not updated_persona:
        raise HTTPException(status_code=500, detail="Failed to update persona")
    
    # Save the updated persona to a file
    try:
        persona_manager.save_persona(updated_persona, format='json')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving persona: {e}")
    
    return PersonaResponse(
        id=updated_persona.id,
        name=updated_persona.name,
        description=updated_persona.description
    )


@router.delete(
    "/{persona_id}",
    response_model=SuccessResponse,
    summary="Delete a persona",
    description="Deletes a persona with the specified ID."
)
async def delete_persona(
    persona_id: str,
    persona_manager=Depends(get_persona_manager)
):
    """Delete a persona."""
    if not persona_manager.delete_persona(persona_id):
        raise HTTPException(status_code=404, detail=f"Persona with ID {persona_id} not found")
    
    return SuccessResponse(message=f"Persona with ID {persona_id} deleted successfully")


@router.post(
    "/upload",
    response_model=PersonaResponse,
    status_code=201,
    summary="Upload a persona file",
    description="Uploads a persona definition file (JSON or YAML) and creates a new persona."
)
async def upload_persona(
    file: UploadFile = File(...),
    persona_manager=Depends(get_persona_manager)
):
    """Upload a persona definition file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided")
    
    # Check file extension
    if not (file.filename.endswith('.json') or file.filename.endswith('.yaml') or file.filename.endswith('.yml')):
        raise HTTPException(status_code=400, detail="File must be a JSON or YAML file")
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        if file.filename.endswith('.json'):
            persona_data = json.loads(content_str)
        else:  # YAML file
            persona_data = yaml.safe_load(content_str)
        
        # Create the persona
        new_persona = persona_manager.add_persona(persona_data)
        
        # Save the persona to a file
        try:
            persona_manager.save_persona(new_persona, format='json')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving persona: {e}")
        
        return PersonaResponse(
            id=new_persona.id,
            name=new_persona.name,
            description=new_persona.description
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except yaml.YAMLError:
        raise HTTPException(status_code=400, detail="Invalid YAML file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}") 