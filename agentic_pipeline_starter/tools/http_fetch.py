"""
HTTP Fetch Tool for Agentic Pipeline

This module implements the HTTP fetch tool that provides safe and configurable
HTTP requests with retry logic, timeout handling, and comprehensive error management.

Author: Agentic Pipeline Team
Date: November 2025
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum
import json

try:
    import aiohttp
    import ssl
except ImportError:
    aiohttp = None
    ssl = None


class HttpMethod(Enum):
    """Supported HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class HttpError(Exception):
    """Custom exception for HTTP-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, url: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.url = url


@dataclass
class HttpRequest:
    """HTTP request configuration."""
    url: str
    method: HttpMethod = HttpMethod.GET
    headers: Optional[Dict[str, str]] = None
    data: Optional[Union[Dict, str, bytes]] = None
    params: Optional[Dict[str, str]] = None
    timeout: Optional[int] = None
    
    def __post_init__(self):
        """Validate request configuration."""
        if not self.url:
            raise ValueError("URL is required for HTTP requests")
        
        if not self.url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")


@dataclass
class HttpResponse:
    """HTTP response container."""
    status_code: int
    data: Any
    headers: Dict[str, str]
    url: str
    execution_time: float
    retry_count: int = 0
    error_message: Optional[str] = None


class HttpFetchTool:
    """
    HTTP fetch tool with retry logic and comprehensive error handling.
    
    Features:
    - Configurable timeout and retry logic
    - Support for multiple HTTP methods
    - Safe SSL handling
    - Request/response validation
    - Comprehensive error handling
    - Async execution with connection pooling
    """
    
    def __init__(
        self,
        default_timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_redirect: int = 10,
        verify_ssl: bool = True
    ):
        """
        Initialize HTTP fetch tool.
        
        Args:
            default_timeout: Default request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
            backoff_factor: Exponential backoff multiplier
            max_redirect: Maximum number of redirects to follow
            verify_ssl: Whether to verify SSL certificates
        """
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.max_redirect = max_redirect
        self.verify_ssl = verify_ssl
        self.logger = logging.getLogger(__name__)
        
        if aiohttp is None:
            self.logger.warning("aiohttp not available. HTTP tool will use fallback implementation")
        
        self.logger.info(
            f"HttpFetchTool initialized: timeout={default_timeout}s, "
            f"max_retries={max_retries}, verify_ssl={verify_ssl}"
        )
    
    async def fetch(self, request: HttpRequest) -> HttpResponse:
        """
        Execute HTTP request with retry logic.
        
        Args:
            request: HTTP request configuration
            
        Returns:
            HTTP response with data and metadata
            
        Raises:
            HttpError: If request fails after all retries
            ValueError: If request configuration is invalid
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(
                    f"HTTP request attempt {attempt + 1}/{self.max_retries + 1}: "
                    f"{request.method.value} {request.url}"
                )
                
                response = await self._execute_request(request)
                response.retry_count = attempt
                response.execution_time = time.time() - start_time
                
                self.logger.info(
                    f"HTTP request successful: {response.status_code} "
                    f"in {response.execution_time:.2f}s"
                )
                
                return response
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"HTTP request attempt {attempt + 1} failed: {str(e)}"
                )
                
                # Don't retry on certain errors
                if isinstance(e, HttpError) and e.status_code in [400, 401, 403, 404]:
                    break
                
                # Wait before retry (except on last attempt)
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** attempt)
                    self.logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        total_time = time.time() - start_time
        error_msg = f"HTTP request failed after {self.max_retries + 1} attempts in {total_time:.2f}s"
        
        if last_error:
            error_msg += f": {str(last_error)}"
        
        self.logger.error(error_msg)
        raise HttpError(error_msg, url=request.url)
    
    async def _execute_request(self, request: HttpRequest) -> HttpResponse:
        """
        Execute a single HTTP request.
        
        Args:
            request: HTTP request configuration
            
        Returns:
            HTTP response
        """
        if aiohttp is not None:
            return await self._execute_with_aiohttp(request)
        else:
            return await self._execute_with_fallback(request)
    
    async def _execute_with_aiohttp(self, request: HttpRequest) -> HttpResponse:
        """
        Execute request using aiohttp.
        
        Args:
            request: HTTP request configuration
            
        Returns:
            HTTP response
        """
        # Prepare SSL context
        ssl_context = None
        if not self.verify_ssl:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        
        # Prepare timeout
        timeout = aiohttp.ClientTimeout(
            total=request.timeout or self.default_timeout
        )
        
        # Prepare connector
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=100,
            ttl_dns_cache=300
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=request.headers or {}
        ) as session:
            
            # Prepare request data
            kwargs = {
                "params": request.params
            }
            
            if request.data:
                if isinstance(request.data, dict):
                    kwargs["json"] = request.data
                elif isinstance(request.data, str):
                    kwargs["data"] = request.data
                else:
                    kwargs["data"] = request.data
            
            # Execute request
            async with session.request(
                request.method.value,
                request.url,
                **kwargs
            ) as response:
                
                # Read response data
                try:
                    # Try to parse as JSON first
                    response_data = await response.json()
                except Exception:
                    # Fallback to text
                    response_data = await response.text()
                
                # Check for HTTP errors
                if response.status >= 400:
                    raise HttpError(
                        f"HTTP {response.status}: {response.reason}",
                        status_code=response.status,
                        url=request.url
                    )
                
                return HttpResponse(
                    status_code=response.status,
                    data=response_data,
                    headers=dict(response.headers),
                    url=str(response.url),
                    execution_time=0.0  # Will be set by caller
                )
    
    async def _execute_with_fallback(self, request: HttpRequest) -> HttpResponse:
        """
        Fallback implementation when aiohttp is not available.
        
        Args:
            request: HTTP request configuration
            
        Returns:
            HTTP response
        """
        import urllib.request
        import urllib.parse
        import urllib.error
        
        self.logger.warning("Using fallback HTTP implementation (urllib)")
        
        # Prepare URL with parameters
        url = request.url
        if request.params:
            url += "?" + urllib.parse.urlencode(request.params)
        
        # Prepare request
        req = urllib.request.Request(url, method=request.method.value)
        
        # Add headers
        if request.headers:
            for key, value in request.headers.items():
                req.add_header(key, value)
        
        # Add data
        if request.data:
            if isinstance(request.data, dict):
                req.data = json.dumps(request.data).encode('utf-8')
                req.add_header('Content-Type', 'application/json')
            elif isinstance(request.data, str):
                req.data = request.data.encode('utf-8')
            else:
                req.data = request.data
        
        try:
            # Execute request in thread pool to make it async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(
                    req, 
                    timeout=request.timeout or self.default_timeout
                )
            )
            
            # Read response
            response_data = response.read().decode('utf-8')
            
            # Try to parse as JSON
            try:
                response_data = json.loads(response_data)
            except json.JSONDecodeError:
                pass  # Keep as string
            
            return HttpResponse(
                status_code=response.getcode(),
                data=response_data,
                headers=dict(response.headers),
                url=response.geturl(),
                execution_time=0.0  # Will be set by caller
            )
            
        except urllib.error.HTTPError as e:
            raise HttpError(
                f"HTTP {e.code}: {e.reason}",
                status_code=e.code,
                url=request.url
            )
        except urllib.error.URLError as e:
            raise HttpError(f"URL error: {str(e)}", url=request.url)
        except Exception as e:
            raise HttpError(f"Request failed: {str(e)}", url=request.url)
    
    def create_request(
        self,
        url: str,
        method: Union[str, HttpMethod] = HttpMethod.GET,
        **kwargs
    ) -> HttpRequest:
        """
        Create HTTP request with validation.
        
        Args:
            url: Target URL
            method: HTTP method
            **kwargs: Additional request parameters
            
        Returns:
            Validated HTTP request
        """
        if isinstance(method, str):
            method = HttpMethod(method.upper())
        
        return HttpRequest(
            url=url,
            method=method,
            **kwargs
        )
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get HTTP tool configuration information.
        
        Returns:
            Tool configuration details
        """
        return {
            "tool_type": "http_fetch",
            "version": "1.0.0",
            "features": [
                "configurable_timeout",
                "retry_logic",
                "ssl_verification",
                "multiple_methods",
                "async_execution"
            ],
            "configuration": {
                "default_timeout": self.default_timeout,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "backoff_factor": self.backoff_factor,
                "verify_ssl": self.verify_ssl
            },
            "supported_methods": [method.value for method in HttpMethod],
            "aiohttp_available": aiohttp is not None
        }