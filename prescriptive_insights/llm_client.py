"""
LLM Client for Ollama Integration

Provides a wrapper around Ollama's HTTP API for generating text
with configurable models and error handling.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, Iterator, Union
import requests
import config

# Configure logging
logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass

class LLMConnectionError(LLMClientError):
    """Exception for connection/transport errors."""
    pass

class LLMGenerationError(LLMClientError):
    """Exception for text generation errors."""
    pass

class LLMClient:
    """Client for interacting with Ollama's HTTP API."""
    
    def __init__(self, 
                 host: str = None,
                 model: str = None,
                 timeout: int = None,
                 max_retries: int = None):
        """Initialize LLM client with configuration."""
        self.host = host or config.OLLAMA_HOST
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self.max_retries = max_retries or config.OLLAMA_MAX_RETRIES
        
        # API endpoints
        self.generate_url = f"{self.host}/api/generate"
        self.health_url = f"{self.host}/api/tags"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "prescriptive-insights/1.0"
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Ollama server is healthy and get available models."""
        try:
            response = self.session.get(
                self.health_url,
                timeout=10  # Shorter timeout for health check
            )
            response.raise_for_status()
            
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            
            return {
                "status": "healthy",
                "host": self.host,
                "models": models,
                "current_model_available": self.model in models
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "unhealthy",
                "host": self.host,
                "error": str(e)
            }
    
    def generate(self, 
                 prompt: str,
                 system_prompt: str = None,
                 options: Dict[str, Any] = None,
                 stream: bool = False) -> Union[str, Iterator[str]]:
        """
        Generate text from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            options: Additional generation options
            stream: Whether to stream the response
            
        Returns:
            Either full text (stream=False) or iterator of text chunks (stream=True)
        """
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Default options
        default_options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        }
        
        if options:
            default_options.update(options)
        
        payload["options"] = default_options
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Retrying LLM generation (attempt {attempt + 1}) after {wait_time}s")
                    time.sleep(wait_time)
                
                response = self.session.post(
                    self.generate_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                if stream:
                    return self._handle_stream_response(response)
                else:
                    return self._handle_sync_response(response)
                    
            except requests.exceptions.Timeout as e:
                last_error = LLMConnectionError(f"Request timeout after {self.timeout}s: {e}")
            except requests.exceptions.ConnectionError as e:
                last_error = LLMConnectionError(f"Connection error: {e}")
            except requests.exceptions.HTTPError as e:
                if response.status_code >= 500:
                    last_error = LLMConnectionError(f"Server error: {e}")
                else:
                    last_error = LLMGenerationError(f"Generation error: {e}")
            except json.JSONDecodeError as e:
                last_error = LLMGenerationError(f"Invalid JSON response: {e}")
            except Exception as e:
                last_error = LLMClientError(f"Unexpected error: {e}")
        
        # All retries failed
        raise last_error
    
    def _handle_sync_response(self, response: requests.Response) -> str:
        """Handle synchronous (non-streaming) response."""
        try:
            data = response.json()
            
            if "response" not in data:
                raise LLMGenerationError("No response field in LLM output")
            
            return data["response"]
            
        except json.JSONDecodeError as e:
            raise LLMGenerationError(f"Invalid JSON in response: {e}")
    
    def _handle_stream_response(self, response: requests.Response) -> Iterator[str]:
        """Handle streaming response."""
        try:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]  # Remove 'data: ' prefix
                    
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError:
                        continue  # Skip malformed JSON chunks
                        
        except Exception as e:
            raise LLMGenerationError(f"Stream processing error: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            health_data = self.health_check()
            
            if health_data["status"] != "healthy":
                return health_data
            
            # Try to get model details
            payload = {"model": self.model}
            response = self.session.post(
                f"{self.host}/api/show",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                model_data = response.json()
                return {
                    "status": "healthy",
                    "host": self.host,
                    "model": self.model,
                    "model_info": model_data
                }
            else:
                return {
                    "status": "healthy",
                    "host": self.host,
                    "model": self.model,
                    "model_info": None,
                    "note": "Model details not available"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "host": self.host,
                "model": self.model,
                "error": str(e)
            }
    
    def close(self):
        """Close the session."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Convenience function for quick usage
def create_llm_client(**kwargs) -> LLMClient:
    """Create an LLM client with default configuration."""
    return LLMClient(**kwargs)

# Test function
def test_connection(host: str = None, model: str = None) -> bool:
    """Test connection to Ollama server."""
    try:
        client = LLMClient(host=host, model=model)
        health = client.health_check()
        client.close()
        return health["status"] == "healthy"
    except Exception:
        return False