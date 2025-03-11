"""WebSocket server for real-time updates.

This module provides a WebSocket server that allows clients to receive
real-time updates about long-running operations, particularly MCP tool usage.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Optional, Set, Any

import websockets
from websockets.server import WebSocketServerProtocol

# Set up logging
logger = logging.getLogger(__name__)

class PersonaWebSocketServer:
    """WebSocket server for persona agent real-time updates.
    
    This class manages WebSocket connections and handles sending
    updates to clients about the progress of long-running operations.
    
    Attributes:
        clients: Dictionary mapping client IDs to WebSocket connections.
        request_clients: Dictionary mapping request IDs to sets of client IDs.
        request_data: Dictionary storing data for active requests.
    """
    
    def __init__(self):
        """Initialize a new WebSocket server."""
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.request_clients: Dict[str, Set[str]] = {}
        self.request_data: Dict[str, Dict[str, Any]] = {}
        self.server = None
        self.logger = logging.getLogger(__name__)
    
    async def register(self, websocket: WebSocketServerProtocol) -> str:
        """Register a new client.
        
        Args:
            websocket: The WebSocket connection.
            
        Returns:
            A unique client ID.
        """
        client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        self.logger.info(f"Client {client_id} connected")
        
        # Send welcome message
        await websocket.send(json.dumps({
            "type": "connection",
            "status": "connected",
            "client_id": client_id
        }))
        
        return client_id
    
    async def unregister(self, client_id: str) -> None:
        """Unregister a client.
        
        Args:
            client_id: The ID of the client to unregister.
        """
        if client_id in self.clients:
            del self.clients[client_id]
            self.logger.info(f"Client {client_id} disconnected")
            
            # Remove client from any request it was subscribed to
            for request_id, clients in list(self.request_clients.items()):
                if client_id in clients:
                    clients.remove(client_id)
                    
                    # If no clients are subscribed to this request, clean it up
                    if not clients:
                        del self.request_clients[request_id]
                        if request_id in self.request_data:
                            del self.request_data[request_id]
    
    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a client connection.
        
        Args:
            websocket: The WebSocket connection.
        """
        client_id = await self.register(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(client_id, data)
                except json.JSONDecodeError:
                    self.logger.warning(f"Received invalid JSON from client {client_id}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON message"
                    }))
        except websockets.ConnectionClosed:
            pass
        finally:
            await self.unregister(client_id)
    
    async def handle_message(self, client_id: str, data: Dict[str, Any]) -> None:
        """Handle a message from a client.
        
        Args:
            client_id: The ID of the client sending the message.
            data: The message data.
        """
        message_type = data.get("type")
        
        if message_type == "subscribe":
            request_id = data.get("request_id")
            if request_id:
                if request_id not in self.request_clients:
                    self.request_clients[request_id] = set()
                self.request_clients[request_id].add(client_id)
                self.logger.info(f"Client {client_id} subscribed to request {request_id}")
                
                # Send any existing data for this request
                if request_id in self.request_data:
                    websocket = self.clients.get(client_id)
                    if websocket:
                        await websocket.send(json.dumps({
                            "type": "update",
                            "request_id": request_id,
                            "data": self.request_data[request_id]
                        }))
        
        elif message_type == "unsubscribe":
            request_id = data.get("request_id")
            if request_id and request_id in self.request_clients:
                if client_id in self.request_clients[request_id]:
                    self.request_clients[request_id].remove(client_id)
                    self.logger.info(f"Client {client_id} unsubscribed from request {request_id}")
                    
                    # If no clients are subscribed to this request, clean it up
                    if not self.request_clients[request_id]:
                        del self.request_clients[request_id]
                        if request_id in self.request_data:
                            del self.request_data[request_id]
    
    async def create_request(self, persona_id: str, message: str) -> str:
        """Create a new request.
        
        Args:
            persona_id: The ID of the persona that will handle the request.
            message: The message for the persona.
            
        Returns:
            A unique request ID.
        """
        request_id = str(uuid.uuid4())
        self.request_data[request_id] = {
            "persona_id": persona_id,
            "message": message,
            "status": "created",
            "progress": []
        }
        self.logger.info(f"Created request {request_id} for persona {persona_id}")
        return request_id
    
    async def update_request(self, request_id: str, status: str, data: Any = None) -> None:
        """Update a request with new status and data.
        
        Args:
            request_id: The ID of the request to update.
            status: The new status of the request.
            data: Optional data to include with the update.
        """
        if request_id not in self.request_data:
            self.logger.warning(f"Tried to update nonexistent request {request_id}")
            return
        
        # Update request data
        self.request_data[request_id]["status"] = status
        if data is not None:
            self.request_data[request_id]["data"] = data
        
        # Add to progress history
        progress_entry = {
            "timestamp": asyncio.get_event_loop().time(),
            "status": status,
        }
        if data:
            progress_entry["data"] = data
        self.request_data[request_id]["progress"].append(progress_entry)
        
        # Notify subscribed clients
        if request_id in self.request_clients:
            for client_id in self.request_clients[request_id]:
                websocket = self.clients.get(client_id)
                if websocket:
                    try:
                        await websocket.send(json.dumps({
                            "type": "update",
                            "request_id": request_id,
                            "status": status,
                            "data": data
                        }))
                    except websockets.ConnectionClosed:
                        await self.unregister(client_id)
    
    async def complete_request(self, request_id: str, result: str) -> None:
        """Mark a request as completed with the final result.
        
        Args:
            request_id: The ID of the request that completed.
            result: The final result of the request.
        """
        await self.update_request(request_id, "completed", result)
        self.logger.info(f"Completed request {request_id}")
        
        # Keep the data for a while, but eventually clean it up
        asyncio.create_task(self._cleanup_request(request_id))
    
    async def fail_request(self, request_id: str, error: str) -> None:
        """Mark a request as failed with an error message.
        
        Args:
            request_id: The ID of the request that failed.
            error: The error message.
        """
        await self.update_request(request_id, "failed", {"error": error})
        self.logger.info(f"Failed request {request_id}: {error}")
        
        # Keep the data for a while, but eventually clean it up
        asyncio.create_task(self._cleanup_request(request_id))
    
    async def _cleanup_request(self, request_id: str, delay: int = 300) -> None:
        """Clean up request data after a delay.
        
        Args:
            request_id: The ID of the request to clean up.
            delay: The delay in seconds before cleaning up.
        """
        await asyncio.sleep(delay)
        
        # Only clean up if no clients are subscribed
        if request_id in self.request_clients and not self.request_clients[request_id]:
            del self.request_clients[request_id]
            if request_id in self.request_data:
                del self.request_data[request_id]
                self.logger.info(f"Cleaned up request {request_id}")
    
    async def start(self, host: str = "localhost", port: int = 8765) -> None:
        """Start the WebSocket server.
        
        Args:
            host: The host to bind to.
            port: The port to bind to.
        """
        self.server = await websockets.serve(self.handle_client, host, port)
        self.logger.info(f"WebSocket server started on ws://{host}:{port}")
    
    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket server stopped")


# Create a global WebSocket server instance
websocket_server = PersonaWebSocketServer()


async def start_websocket_server(host: str = "localhost", port: int = 8765) -> None:
    """Start the WebSocket server.
    
    Args:
        host: The host to bind to.
        port: The port to bind to.
    """
    await websocket_server.start(host, port)


async def stop_websocket_server() -> None:
    """Stop the WebSocket server."""
    await websocket_server.stop()
