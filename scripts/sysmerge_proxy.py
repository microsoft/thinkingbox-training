# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import math
import os
from contextlib import asynccontextmanager
from copy import deepcopy
from urllib.parse import urlsplit

import anyio
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

logger = logging.getLogger("uvicorn.error")
REQUEST_HEADERS = {"authorization", "api-key", "accept", "content-type"}
RESPONSE_HEADERS = {"content-type", "retry-after", "x-request-id"}


def _validate_system_content(content):
    if isinstance(content, str):
        return
    if isinstance(content, list) and all(
        isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
        for part in content
    ):
        return
    raise ValueError("System content must be a string or an array of text parts")


def merge_leading_system(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    if any(not isinstance(message, dict) for message in messages):
        raise ValueError("Every message must be an object")

    prefix_end = 0
    while prefix_end < len(messages) and messages[prefix_end].get("role") == "system":
        _validate_system_content(messages[prefix_end].get("content"))
        prefix_end += 1
    if any(message.get("role") == "system" for message in messages[prefix_end:]):
        raise ValueError("Qwen requires system messages to form one leading prefix")

    normalized = deepcopy(messages)
    if prefix_end <= 1:
        return normalized
    if any(set(message) - {"role", "content"} for message in messages[:prefix_end]):
        raise ValueError("Cannot merge system messages with additional message fields")

    contents = [message["content"] for message in normalized[:prefix_end]]
    if all(isinstance(content, str) for content in contents):
        merged_content = "\n\n".join(contents)
    else:
        merged_content = []
        for index, content in enumerate(contents):
            if index:
                merged_content.append({"type": "text", "text": "\n\n"})
            if isinstance(content, str):
                merged_content.append({"type": "text", "text": content})
            else:
                merged_content.extend(content)
    return [{"role": "system", "content": merged_content}, *normalized[prefix_end:]]


def _reject_nonfinite(value):
    raise ValueError("Non-finite numbers are not valid JSON")


def _parse_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite numbers are not valid JSON")
    return number


def _error(status_code, message, error_type):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


def _response_headers(response):
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() in RESPONSE_HEADERS
    }


def create_app(backend, *, transport=None):
    parsed = urlsplit(backend)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "BACKEND must be an HTTP(S) origin without credentials or a path"
        )

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            base_url=backend,
            timeout=httpx.Timeout(1800.0, connect=30.0),
            follow_redirects=False,
            transport=transport,
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    async def forward(request, *, payload=None, stream=False):
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() in REQUEST_HEADERS
        }
        kwargs = (
            {"json": payload}
            if payload is not None
            else {"content": await request.body()}
        )
        if payload is not None:
            headers["content-type"] = "application/json"
        client = request.app.state.client
        upstream = client.build_request(
            request.method,
            request.url.path,
            params=list(request.query_params.multi_items()),
            headers=headers,
            **kwargs,
        )
        try:
            response = await client.send(upstream, stream=stream)
            if stream and response.is_success:

                async def chunks():
                    try:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    except httpx.RequestError as exc:
                        logger.warning(
                            "Qwen proxy upstream stream failed: %s", type(exc).__name__
                        )
                        raise
                    finally:
                        with anyio.CancelScope(shield=True):
                            await response.aclose()

                return StreamingResponse(
                    chunks(),
                    status_code=response.status_code,
                    headers=_response_headers(response),
                    background=BackgroundTask(response.aclose),
                )
            try:
                await response.aread()
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=_response_headers(response),
                )
            finally:
                await response.aclose()
        except httpx.TimeoutException:
            logger.warning("Qwen proxy upstream request timed out")
            return _error(504, "Upstream request timed out", "upstream_timeout")
        except httpx.RequestError as exc:
            logger.warning(
                "Qwen proxy upstream connection failed: %s", type(exc).__name__
            )
            return _error(
                502, "Upstream connection failed", "upstream_connection_error"
            )

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready(request: Request):
        try:
            response = await request.app.state.client.get("/v1/models", timeout=5.0)
        except httpx.RequestError as exc:
            logger.warning("Qwen proxy readiness failed: %s", type(exc).__name__)
            return _error(503, "Upstream is not ready", "upstream_not_ready")
        if not response.is_success:
            logger.warning(
                "Qwen proxy readiness returned HTTP %s", response.status_code
            )
            return _error(503, "Upstream is not ready", "upstream_not_ready")
        return {"ready": True}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            payload = json.loads(
                await request.body(),
                parse_constant=_reject_nonfinite,
                parse_float=_parse_float,
            )
            if not isinstance(payload, dict):
                raise TypeError("Request body must be a JSON object")
            if not isinstance(payload.get("stream", False), bool):
                raise TypeError("stream must be a boolean")
            payload["messages"] = merge_leading_system(payload.get("messages"))
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            logger.warning("Qwen proxy rejected request: %s", exc)
            return _error(400, str(exc), "invalid_request_error")
        return await forward(
            request, payload=payload, stream=payload.get("stream", False)
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PUT"])
    async def passthrough(path: str, request: Request):
        return await forward(request)

    return app


app = create_app(os.environ.get("BACKEND", "http://127.0.0.1:8001"))
