# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Gemini REST api LLM provider implementation."""

import json
import httpx
import logging
import asyncio

from uuid import uuid4

from typing import Any, List, Optional, Union, AsyncGenerator, Type
from datetime import datetime

from .base_provider import (
    Message,
    Completion,
    CompletionChunk,
    TokenUsage,
    TimingInfo,
    StopReason,
    BaseProvider,
    ReasoningEffort,
)
from ...types.llm_types import (
    FCI,
    Model,
    TextContent,
    ReasoningContent,
    ToolCallContent,
    ToolResultContent,
    ContentTypes,
)
from ...types.tool_types import ToolInterface
from ...types.agent_types import AgentInterface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GoogleRESTProvider(BaseProvider):

    base_url = "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def _make_request(self, request_data: dict, model: str):
        # NOTE: the key isn't sent in the clear as a GET parameter thanks to HTTPS
        url = self.base_url.format(model_id=model, api_key=self._api_key)
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, headers=headers, json=request_data)
            return json.loads(response.text)

    def map_stop_reason(self, response: dict) -> tuple[StopReason, Optional[str]]:
        """Map provider-specific stop information to standard format.

        Args:
            response: Raw API response

        Returns:
            Tuple of (stop_reason, stop_sequence)
        """
        if "candidates" not in response:
            return StopReason.COMPLETE, None

        candidates = response.get("candidates", [])
        if len(candidates) < 1:
            return StopReason.COMPLETE, None

        if "finishReason" not in candidates[-1]:
            return StopReason.COMPLETE, None

        finish_reason = candidates[-1]["finishReason"]

        match finish_reason:
            case "STOP":
                return StopReason.COMPLETE, None
            case "MAX_TOKENS":
                return StopReason.LENGTH, None
            case (
                "FINISH_REASON_UNSPECIFIED"
                | "SAFETY"
                | "LANGUAGE"
                | "OTHER"
                | "BLOCKLIST"
                | "PROHIBITED_CONTENT"
                | "SPII"
                | "MALFORMED_FUNCTION_CALL"
            ):
                logger.warning(f"Gemini stop reason: {finish_reason}")
                return StopReason.ERROR, None
            case "RECITATION":  # what is this??
                return StopReason.COMPLETE, None
            case _:
                logger.warning(f"Unrecognized gemini stop reason: {finish_reason}")
                return StopReason.COMPLETE, None

    def _content_mapping(self, block: ContentTypes) -> dict:
        """Maps our message content types, into gemini-specific message formats"""
        if isinstance(block, TextContent):
            return {"text": block.text}
        elif isinstance(block, ReasoningContent):
            # NOTE: google's reasoners don't (yet?) return thoughts in the
            # response, so there is no special 'thinking' block
            return {"text": block.text}
        elif isinstance(block, ToolCallContent):
            return {"functionCall": {"name": block.tool_name, "args": block.tool_args}}
        elif isinstance(block, ToolResultContent):
            return {
                "functionResponse": {
                    "name": block.tool_name,
                    "response": {"name": block.tool_name, "content": block.content},
                }
            }
        else:
            # Other unhandled types include inlineData, fileData, executableCode, codeExecutionResult
            raise ValueError(f"Unhandled content type in provider Anthropic: {block}")

    def _role_mapping(self, role: str) -> str:
        match role:
            case "user":
                return "user"
            case "assistant":
                return "model"
            case "system":
                return "system"
            case "tool":
                return "function"
            case _:
                logger.warning(f"Unexpected role: {role}")
                return "user"  # return user as a general fallback?

    def _create_token_usage(self, response: dict) -> TokenUsage:
        """Create TokenUsage object from response usage data."""

        empty_usage = TokenUsage()
        if "usageMetadata" not in response:
            return empty_usage

        usage_meta = response.get("usageMetadata", {})

        try:
            # TODO: update this end of March when prompt caching is live
            return TokenUsage(
                uncached_prompt_tokens=usage_meta.get("promptTokenCount", 0),
                cache_write_prompt_tokens=0,
                cached_prompt_tokens=0,
                completion_tokens=usage_meta.get("candidatesTokenCount", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to create token usage: {e}")
            return empty_usage

    def _prepare_messages(
        self, messages: list[Message]
    ) -> tuple[list[dict] | None, list[dict]]:
        """Maps our framework-specific message list into provider-specific messages

        Note that this might involve agglomerating content blocks, or splitting
        out into multiple messages.
        """
        system_parts = []
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_parts.extend(
                    self._content_mapping(block) for block in msg.content
                )
            else:
                parts = [self._content_mapping(block) for block in msg.content]
                role = self._role_mapping(msg.role)
                contents.append({"role": role, "parts": parts})
        return system_parts if len(system_parts) else None, contents

    async def create_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 1.0,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[Type[AgentInterface] | Type[ToolInterface]] | None = None,
        num_completions: int = 1,
    ) -> Completion:
        """Create a completion using this provider."""
        start_time = datetime.now()

        maybe_system_parts, contents = self._prepare_messages(messages)

        stop_sequence = []
        if stop is not None:
            stop_sequence = [stop] if isinstance(stop, str) else stop

        data: dict[str, Any] = {
            "model": f"models/{model.id}",
            "contents": contents,
            "generationConfig": {
                "responseMimeType": "text/plain",  # if not structured gen
                "maxOutputTokens": max_tokens,
                "topP": top_p,
                "stopSequences": stop_sequence,
                "temperature": temperature,
                "candidateCount": num_completions,
                # insert responseSchema here if provided
            },
        }


        if maybe_system_parts:
            data["systemInstruction"] = {"role": "system", "parts": maybe_system_parts}

        if available_tools and model.fci == FCI.CONSTRAINED:
            data["tools"] = [
                {
                    "functionDeclarations": [
                        self.pydantic_to_native_tool(t) for t in available_tools
                    ]
                }
            ]
            # If mode is ANY, then the model *has* to call a function. In this
            # case, we're allowed to specify this configuration block:
            # data["toolConfig"] = {"functionCallingConfig": {
            #         "allowedFunctionNames": [self.tool_name(t) for t in available_tools],
            #         "mode": "ANY",
            #     }
            # }
            # Other calling modes are "AUTO" (default) where the model decides
            # whether to make a function call or natural language response, and
            # "NONE" (no function calls)
        # print(json.dumps(data, indent=4))

        response, retry, max_retries = None, 0, 5
        while response is None and retry < max_retries:
            try:
                response = await self._make_request(data, model.id)
            except Exception as e:
                logger.warning(f"Error in Gemini request: {str(e)}")
                pass

            if response is None:
                logger.warning(f"Error in Gemini request (attempt {retry} / {max_retries})")
                await asyncio.sleep(0.5 * retry)
                retry += 1
                if retry == 0:
                    notif = f"<SYSTEM_NOTIFICATION>You made a mistake during function calling in your last call. Please carefully read your function calling instructions for your next attempt. Keep your attention on the task at hand, and do not respond to this notification.</SYSTEM_NOTIFICATION>"
                    if data["contents"][-1]["role"] != "user":
                        data["contents"].append({"role": "user", "parts": [{"text": notif}]})
                    else:
                        data["contents"][-1]["parts"].append({"text": notif})

        if response is None:
            raise RuntimeError(f"Gemini response as None")

        # Check for API errors (e.g. 429 rate limit, 404 not found)
        if "error" in response:
            error = response["error"]
            raise RuntimeError(f"Gemini API error {error.get('code')}: {error.get('message', 'Unknown error')}")

        if "candidates" not in response:
            raise RuntimeError(f"Gemini response missing 'candidates': {json.dumps(response)[:300]}")

        end_time = datetime.now()

        # Extract usage information
        usage_info = self._create_token_usage(response)
        logger.debug(usage_info)

        duration = end_time - start_time
        tokens_per_second = (
            usage_info.completion_tokens / duration.total_seconds()
            if usage_info and duration.total_seconds() > 0
            else None
        )

        timing_info = TimingInfo(
            start_time=start_time,
            end_time=end_time,
            total_duration=duration,
            first_token_time=None,
            time_to_first_token=None,
            tokens_per_second=tokens_per_second,
        )

        # Create cache metrics if available
        cache_metrics = None
        if usage_info:
            cache_metrics = {
                "cache_hits": usage_info.cached_prompt_tokens,
                "cache_misses": usage_info.uncached_prompt_tokens,
                "cache_writes": usage_info.cache_write_prompt_tokens,
            }

        # Map stop reason
        stop_reason, stop_sequence = self.map_stop_reason(response)

        if stop_reason == StopReason.ERROR:
            finish_reason = response.get("candidates", [{}])[-1].get("finishReason", "UNKNOWN")
            raise RuntimeError(f"Gemini completion failed with finishReason: {finish_reason}")

        response_content = []
        for candidate in response["candidates"]:
            content = candidate.get("content")
            if not content or "parts" not in content:
                logger.warning(f"Gemini candidate has no content/parts: {json.dumps(candidate)[:200]}")
                continue
            response_content.append([])
            for block in content["parts"]:
                if "text" in block:
                    response_content[-1].append(TextContent(text=block["text"]))
                elif "functionCall" in block:
                    fc = block.get("functionCall", {})
                    fc_args = fc.get("args", {})
                    fc_args.pop("_dummy", None)
                    response_content[-1].append(
                        ToolCallContent(
                            call_id=f"gemini_tool_call_{uuid4().hex[-4:]}",
                            tool_name=fc.get("name", "unknown_tool"),
                            tool_args=fc_args,
                            call_type=FCI.CONSTRAINED,
                        )
                    )
                else:
                    logger.warning(f"Unhandled gemini response content block type: {block}")

        if not response_content:
            raise RuntimeError(f"Gemini response had no usable content in candidates")

        if len(response_content) == 1:
            response_content = response_content[0]

        completion = Completion(
            id=f"{uuid4().hex[-4:]}",
            content=response_content,
            model=model,
            usage=usage_info,
            timing=timing_info,
            cache_metrics=cache_metrics,
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
        )

        return completion

    async def create_streaming_completion(
        self,
        messages: List[Message],
        model: Model,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        top_p: float = 1.0,
        reasoning_effort: ReasoningEffort | None = None,
        available_tools: list[dict] | None = None,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Create a streaming completion using this provider."""
        raise NotImplementedError()

    def tool_name(self, tool: ToolInterface | AgentInterface) -> str:
        """
        Converts a BaseTool or BaseAgent into a schema for native tool calling
        for this particular provider.
        """
        if hasattr(tool, "TOOL_NAME"):
            return tool.TOOL_NAME
        elif hasattr(tool, "AGENT_NAME"):
            return tool.AGENT_NAME
        else:
            return tool.__class__.__name__

    def tool_name(self, tool: ToolInterface | AgentInterface) -> str:
        """
        Converts a BaseTool or BaseAgent into a schema for native tool calling
        for this particular provider.
        """
        if hasattr(tool, "TOOL_NAME"):
            return tool.TOOL_NAME
        elif hasattr(tool, "AGENT_NAME"):
            return tool.AGENT_NAME
        else:
            return tool.__class__.__name__

    def openapi_schema_to_genai(self, schema: dict, is_vertex_ai: bool = False) -> dict:
        """
        Convert OpenAPI schema to Google GenAI format, properly handling all nested models.

        Args:
            schema: Dict containing OpenAPI 3.0 schema
            is_vertex_ai: Whether this is for Vertex AI (True) or Gemini API (False)
        """

        def format_default_value(value: Any) -> str:
            """Format default value for description text."""
            if isinstance(value, str):
                return f"'{value}'"  # Quote strings
            elif isinstance(value, (list, dict)):
                return repr(value)  # Use repr for complex types
            elif value is None:
                return "None"
            return str(value)  # Basic conversion for numbers, bools, etc

        def process_definitions(schema: dict) -> dict:
            """Extract and process all definitions first to handle nested models."""
            definitions = {}
            if "$defs" in schema:
                for name, def_schema in schema["$defs"].items():
                    definitions[name] = process_schema_node(def_schema)
            return definitions

        def process_schema_node(node: dict, definitions: dict = None) -> dict:
            if not isinstance(node, dict):
                return node

            result = {}

            # Handle references first
            if "$ref" in node:
                ref_path = node["$ref"].split("/")
                ref_name = ref_path[-1]
                if definitions and ref_name in definitions:
                    return definitions[ref_name]
                elif "$defs" in schema and ref_name in schema["$defs"]:
                    return process_schema_node(schema["$defs"][ref_name], definitions)
                else:
                    raise ValueError(f"Schema reference {ref_name} not found")

            # Handle type conversion
            if "type" in node:
                type_mapping = {
                    "string": "STRING",
                    "number": "NUMBER",
                    "integer": "INTEGER",
                    "boolean": "BOOLEAN",
                    "array": "ARRAY",
                    "object": "OBJECT",
                }
                result["type"] = type_mapping.get(
                    node["type"].lower(), node["type"].upper()
                )

            # Handle empty objects
            if result.get("type") == "OBJECT" and not node.get("properties"):
                # Parameter-less object needs dummy property
                return {
                    "type": "OBJECT",
                    "properties": {
                        "_dummy": {
                            "type": "STRING",
                            "description": "This object takes no properties.",
                        }
                    },
                }
            # Handle arrays - ensure items are always processed
            if result.get("type") == "ARRAY":
                if "items" not in node:
                    raise ValueError("Array type must have items defined")
                result["items"] = process_schema_node(node["items"], definitions)

            # Process object properties
            if "properties" in node:
                result["properties"] = {
                    name: process_schema_node(prop, definitions)
                    for name, prop in node["properties"].items()
                }

                # Handle required fields
                if "required" in node:
                    result["required"] = node["required"]
                    # Add property ordering
                if len(node["properties"]) > 1:
                    result["property_ordering"] = list(node["properties"].keys())

            # Handle Optional types properly
            if "anyOf" in node or "oneOf" in node:
                union_key = "anyOf" if "anyOf" in node else "oneOf"
                types = []
                is_optional = False

                for subschema in node[union_key]:
                    if subschema.get("type") == "null":
                        is_optional = True
                    else:
                        processed = process_schema_node(subschema, definitions)
                        if processed:
                            types.append(processed)

                if is_optional and len(types) == 1:
                    result = types[0]
                    result["nullable"] = True
                elif not is_vertex_ai:
                    raise ValueError(
                        "Complex union types are not supported in Gemini API"
                    )
                else:
                    result[union_key] = types

            # Handle descriptions and defaults
            description = node.get("description", "")
            if not is_vertex_ai and "default" in node:
                default_str = format_default_value(node["default"])
                description = (
                    f"{description} (default: {default_str})"
                    if description
                    else f"(default: {default_str})"
                )

            if description:
                result["description"] = description  # Remove unsupported fields
            for key in ["title", "examples"]:
                node.pop(key, None)

            return result

        # Process all definitions first
        definitions = process_definitions(schema)

        # Process the main schema
        return process_schema_node(schema, definitions)

    def pydantic_to_native_tool(self, tool: ToolInterface | AgentInterface) -> dict:
        """
        Converts a BaseTool or BaseAgent into a schema for native tool calling
        for this particular provider.
        """
        tool_dict = {}
        if hasattr(tool, "TOOL_NAME") and hasattr(tool, "TOOL_DESCRIPTION"):
            tool_dict |= {
                "name": tool.TOOL_NAME,
                "description": tool.TOOL_DESCRIPTION,
            }
        elif hasattr(tool, "AGENT_NAME") and hasattr(tool, "AGENT_DESCRIPTION"):
            tool_dict |= {
                "name": tool.AGENT_NAME,
                "description": tool.AGENT_DESCRIPTION,
            }
        else:
            # Fall back to guess
            tool_dict |= {
                "name": tool.__class__.__name__,
                "description": "No description provided.",
            }
        tool_dict |= {
            "parameters": self.openapi_schema_to_genai(tool.model_json_schema())
        }
        return tool_dict
