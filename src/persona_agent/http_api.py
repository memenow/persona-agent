"""HTTP API for Persona Agent.

This module provides a Flask-based HTTP API for the persona agent service.
"""

import asyncio
import json
import logging
import os
import traceback
import threading
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from persona_agent.api import PersonaAgentAPI
from persona_agent.core.persona_profile import PersonaProfile
from persona_agent.llm_config import config_manager
from persona_agent.websocket_server import websocket_server, start_websocket_server

# Set up logging
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/v1/*": {"origins": "*"}})  # Enable CORS for all routes with proper configuration

# Create a global PersonaAgentAPI instance
api = PersonaAgentAPI()

# Store active requests
active_requests = {}

# Start the WebSocket server in a separate thread
def start_websocket_server_thread(host="localhost", port=8765):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_websocket_server(host, port))
    loop.run_forever()

websocket_thread = threading.Thread(
    target=start_websocket_server_thread,
    daemon=True
)
websocket_thread.start()
logger.info("Started WebSocket server thread")


def handle_error(e: Exception) -> Tuple[Dict[str, Any], int]:
    """Handle exceptions and return appropriate HTTP responses.
    
    Args:
        e: The exception to handle.
        
    Returns:
        A tuple containing the error response and HTTP status code.
    """
    logger.error(f"API error: {str(e)}\n{traceback.format_exc()}")
    
    if isinstance(e, HTTPException):
        return {"error": str(e), "type": e.__class__.__name__}, e.code
    
    # Map common exceptions to HTTP status codes
    status_code = 500  # Default to internal server error
    if isinstance(e, ValueError):
        status_code = 400  # Bad request
    elif isinstance(e, KeyError):
        status_code = 404  # Not found
    elif isinstance(e, PermissionError):
        status_code = 403  # Forbidden
    
    return {"error": str(e), "type": e.__class__.__name__}, status_code


@app.route("/v1/personas", methods=["GET"])
def list_personas() -> Response:
    """List all available personas.
    
    Returns:
        JSON response with the list of personas.
    """
    try:
        personas = api.list_personas()
        return jsonify({"personas": personas})
    except Exception as e:
        error_response, status_code = handle_error(e)
        return jsonify(error_response), status_code


@app.route("/v1/personas", methods=["POST"])
def create_persona() -> Response:
    """Create a new persona.

    Request body:
        - profile (dict or string): Persona profile data or path to a profile file
        - profile_data (dict, optional): Complete persona profile data structure 
        - persona_id (optional): Unique identifier for the persona
        - model_name (optional): Name of the LLM model to use
        - enable_mcp_tools (optional): Whether to enable MCP tools

    Returns:
        JSON response with the created persona ID.
    """
    try:
        data = request.json

        if not data:
            return jsonify({"error": "Missing request body"}), 400

        # Check if profile data is directly provided
        profile_data = data.get("profile_data")
        if profile_data:
            # Use directly provided profile data
            profile = PersonaProfile.from_dict(profile_data)
        else:
            # Fall back to using profile path or dict
            profile = data.get("profile")
            if not profile:
                return jsonify({"error": "Missing required field: either profile or profile_data"}), 400

            # Convert profile to PersonaProfile if it's a dictionary
            if isinstance(profile, dict):
                profile = PersonaProfile.from_dict(profile)

        # Get LLM configuration based on the requested model
        model_name = data.get("model_name")
        try:
            model_config = config_manager.get_model_config(model_name)
            llm_client = config_manager.get_llm_client(model_name)
            llm_config = {"client": llm_client}
        except Exception as e:
            logger.warning(f"Failed to get LLM client: {str(e)}")
            llm_config = None

        # Create the persona
        persona_id = api.create_persona(
            profile=profile,
            persona_id=data.get("persona_id"),
            llm_config=llm_config,
            enable_mcp_tools=data.get("enable_mcp_tools", True),
        )

        return jsonify({"persona_id": persona_id})
    except Exception as e:
        error_response, status_code = handle_error(e)
        return jsonify(error_response), status_code


@app.route("/v1/personas/<persona_id>", methods=["GET"])
def get_persona(persona_id: str) -> Response:
    """Get information about a specific persona.
    
    Args:
        persona_id: ID of the persona to get.
    
    Returns:
        JSON response with the persona information.
    """
    try:
        persona = api.get_persona(persona_id)
        
        # Extract persona information
        persona_info = {
            "id": persona_id,
            "name": persona.profile.name,
            "description": persona.profile.description,
            "personal_background": persona.profile.personal_background,
            "language_style": persona.profile.language_style,
            "knowledge_domains": persona.profile.knowledge_domains,
        }
        
        return jsonify(persona_info)
    except Exception as e:
        error_response, status_code = handle_error(e)
        return jsonify(error_response), status_code


@app.route("/v1/personas/<persona_id>", methods=["DELETE"])
def delete_persona(persona_id: str) -> Response:
    """Delete a specific persona.
    
    Args:
        persona_id: ID of the persona to delete.
    
    Returns:
        JSON response indicating successful deletion.
    """
    try:
        api.delete_persona(persona_id)
        return jsonify({"message": f"Persona {persona_id} deleted successfully"})
    except Exception as e:
        error_response, status_code = handle_error(e)
        return jsonify(error_response), status_code


@app.route("/v1/personas/<persona_id>/chat", methods=["POST"])
def chat_with_persona(persona_id: str) -> Response:
    """Chat with a specific persona.
    
    Args:
        persona_id: ID of the persona to chat with.
    
    Request body:
        - message (str): Message to send to the persona
    
    Returns:
        JSON response with the persona's reply.
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        
        message = data.get("message")
        if not message:
            return jsonify({"error": "Missing required field: message"}), 400
        
        # Use WebSockets for all queries to allow the LLM to decide when to use tools
        # This approach lets the model determine tool usage based on context rather than keywords
        
        # Create a unique request ID
        request_id = data.get("request_id") or str(uuid.uuid4())
        
        # Create a WebSocket request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        websocket_request_id = loop.run_until_complete(
            websocket_server.create_request(persona_id, message)
        )
        
        # Send initial notification via WebSocket
        initial_message = "Processing your request..."
        loop.run_until_complete(
            websocket_server.update_request(websocket_request_id, "processing", initial_message)
        )
        
        # Start a background task to process the request
        def process_chat_async():
            try:
                # This will run in the background
                # Need a new event loop in this thread
                thread_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(thread_loop)
                
                # Process the chat and get the response
                # Pass the request_id to track progress
                # Get the persona first and check if it exists
                persona = api.get_persona(persona_id)
                if persona is None:
                    raise ValueError(f"Persona {persona_id} not found or not properly initialized")
                
                # Then call _async_chat with the request_id
                final_response = thread_loop.run_until_complete(
                    persona._async_chat(message, websocket_request_id)
                )
                
                # Send the final response via WebSocket
                thread_loop.run_until_complete(
                    websocket_server.complete_request(websocket_request_id, final_response)
                )
                
                logger.info(f"Completed async chat for persona {persona_id}, request {websocket_request_id}")
            except Exception as e:
                # Send error via WebSocket
                try:
                    error_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(error_loop)
                    error_loop.run_until_complete(
                        websocket_server.fail_request(websocket_request_id, str(e))
                    )
                except Exception as ws_error:
                    logger.error(f"Error sending WebSocket error: {str(ws_error)}")
                
                logger.error(f"Error in async chat: {str(e)}")
        
        # Start the background task
        background_thread = threading.Thread(target=process_chat_async)
        background_thread.daemon = True
        background_thread.start()
        
        # Return an immediate response
        return jsonify({
            "persona_id": persona_id,
            "request_id": websocket_request_id,
            "response": initial_message,
            "status": "processing"
        })
    except Exception as e:
        error_response, status_code = handle_error(e)
        return jsonify(error_response), status_code


@app.route("/v1/personas/<persona_id>/tools", methods=["GET"])
def list_persona_tools(persona_id: str) -> Response:
    """List the tools available to a specific persona.
    
    Args:
        persona_id: ID of the persona.
    
    Returns:
        JSON response with the list of available tools.
    """
    try:
        persona = api.get_persona(persona_id)
        
        if not hasattr(persona, "tool_adapter"):
            return jsonify({"tools": []})
        
        # Get available tools
        tools = persona.tool_adapter.get_available_tools()
        tool_configs = persona.tool_adapter.get_tool_configs()
        
        return jsonify({
            "tools": tools,
            "tool_configs": tool_configs,
        })
    except Exception as e:
        error_response, status_code = handle_error(e)
        return jsonify(error_response), status_code


@app.route("/v1/personas/<persona_id>/tools/<tool_id>", methods=["POST"])
def execute_persona_tool(persona_id: str, tool_id: str) -> Response:
    """Execute a tool for a specific persona.
    
    Args:
        persona_id: ID of the persona.
        tool_id: ID of the tool to execute.
    
    Request body:
        - arguments: Dictionary of arguments for the tool.
    
    Returns:
        JSON response with the tool execution result.
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        
        arguments = data.get("arguments", {})
        
        # Get the persona
        persona = api.get_persona(persona_id)
        
        if not hasattr(persona, "tool_adapter"):
            return jsonify({"error": "Persona does not have tools enabled"}), 400
        
        # Check if the tool exists
        available_tools = persona.tool_adapter.get_available_tools()
        if tool_id not in available_tools:
            return jsonify({"error": f"Tool {tool_id} not found for persona {persona_id}"}), 404
        
        # Get the tool function
        tool_function = persona.tool_adapter.function_map.get(tool_id)
        if not tool_function:
            return jsonify({"error": f"Tool function {tool_id} not found"}), 404
        
        # Execute the tool
        import asyncio
        result = asyncio.run(tool_function(**arguments))
        
        # Log the tool execution
        logger.info(f"Executed tool {tool_id} for persona {persona_id} with arguments: {arguments}")
        logger.info(f"Tool execution result: {result}")
        
        return jsonify({
            "persona_id": persona_id,
            "tool_id": tool_id,
            "arguments": arguments,
            "result": result,
        })
    except Exception as e:
        error_response, status_code = handle_error(e)
        return jsonify(error_response), status_code


@app.route("/v1/personas/<persona_id>/save", methods=["POST"])
def save_persona(persona_id: str) -> Response:
    """Save a specific persona to a file.
    
    Args:
        persona_id: ID of the persona to save.
    
    Request body:
        - file_path (str): Path where to save the persona
    
    Returns:
        JSON response indicating successful saving.
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        
        file_path = data.get("file_path")
        if not file_path:
            return jsonify({"error": "Missing required field: file_path"}), 400
        
        # Save the persona
        api.save_persona(persona_id, file_path)
        
        return jsonify({
            "message": f"Persona {persona_id} saved to {file_path}",
            "file_path": file_path,
        })
    except Exception as e:
        error_response, status_code = handle_error(e)
        return jsonify(error_response), status_code


@app.route("/v1/personas/load", methods=["POST"])
def load_persona() -> Response:
    """Load a persona from a file.
    
    Request body:
        - file_path (str): Path to load the persona from
        - persona_id (optional): ID to assign to the loaded persona
        - model_name (optional): Name of the LLM model to use
        - enable_mcp_tools (optional): Whether to enable MCP tools
    
    Returns:
        JSON response with the loaded persona ID.
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        
        file_path = data.get("file_path")
        if not file_path:
            return jsonify({"error": "Missing required field: file_path"}), 400
        
        # Get LLM configuration based on the requested model
        model_name = data.get("model_name")
        try:
            model_config = config_manager.get_model_config(model_name)
            llm_client = config_manager.get_llm_client(model_name)
            llm_config = {"client": llm_client}
        except Exception as e:
            logger.warning(f"Failed to get LLM client: {str(e)}")
            llm_config = None
        
        # Load the persona
        persona_id = api.load_persona_from_file(
            file_path=file_path,
            persona_id=data.get("persona_id"),
            llm_config=llm_config,
            enable_mcp_tools=data.get("enable_mcp_tools", True),
        )
        
        return jsonify({"persona_id": persona_id})
    except Exception as e:
        error_response, status_code = handle_error(e)
        return jsonify(error_response), status_code


@app.route("/v1/models", methods=["GET"])
def list_models() -> Response:
    """List all available LLM models.
    
    Returns:
        JSON response with the list of available models.
    """
    try:
        models = list(config_manager.model_configs.keys())
        default_model = config_manager.default_model
        
        return jsonify({
            "models": models,
            "default_model": default_model,
        })
    except Exception as e:
        error_response, status_code = handle_error(e)
        return jsonify(error_response), status_code


@app.route("/v1/health", methods=["GET"])
def health_check() -> Response:
    """Check the health of the API.
    
    Returns:
        JSON response indicating the API is healthy.
    """
    return jsonify({
        "status": "ok",
        "version": "0.2.0",
    })


def create_app() -> Flask:
    """Create and configure the Flask application.
    
    Returns:
        The configured Flask application.
    """
    # Additional configuration can be applied here
    return app


def run_server(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    """Run the HTTP API server.
    
    Args:
        host: The host to run the server on.
        port: The port to run the server on.
        debug: Whether to run the server in debug mode.
    """
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Run the server
    run_server(debug=True)
