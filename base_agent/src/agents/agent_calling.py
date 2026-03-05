# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for making agent calls"""

import json
import asyncio
import logging

from datetime import datetime

from .base_agent import agent_registry
from ..events import EventBus
from ..utils.parsing import extract_between_patterns
from ..schemas import dumps
from ..callgraph.manager import CallGraphManager
from ..utils.stop_tokens import AGENT_STOP_TOKEN
from ..types.agent_types import AgentStatus, AgentMetrics, AgentResult, AgentInterface
from ..types.event_types import EventType, Event
from ..types.llm_types import FCI, ToolCallContent
from ..types.common import ArgFormat


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_agent_instructions(arg_format: ArgFormat = ArgFormat.XML) -> str:
    """Get instructions to add into the system prompt about how to call agents"""

    if arg_format == ArgFormat.JSON:
        instructions = f"""When you need to call a agent, format your request using a valid JSON string inside this XML like structure:

<AGENT_CALL>
<AGENT_NAME>agent_name</AGENT_NAME>
<AGENT_ARGS>
{{
    "foo": 1,
    "bar": "value"
}}
</AGENT_ARGS>
{AGENT_STOP_TOKEN}
"""
    else:
        instructions = f"""When you need to call a agent, format your request as follows, with the tool arguments all inside the TOOL_ARGS tag, and the case-sensitive argument names within their own tags:

<AGENT_CALL>
<AGENT_NAME>tool_name</AGENT_NAME>
<AGENT_ARGS>
<arg1>value1</arg1>
<arg2>value2</arg2>
</AGENT_ARGS>
{AGENT_STOP_TOKEN}
"""

    instructions += """\n\nRemember, tool calls are different from agent calls, and to invoke the latter you must use <AGENT_CALL> and not <TOOL_CALL>. Using an agent name in a TOOL_CALL will result an in error (and vice versa).

Further, do not wrap the agent call in backticks like a code block. Simply start generating the <AGENT_CALL> block directly.
"""

    return instructions


async def parse_agent_content(
    agent_content: str, arg_format: ArgFormat
) -> tuple[str | None, dict | None, str | None]:
    """
    For unconstrained function calling, parse the extracted plain-text agent
    call content, and attempt to return the agent name and the dictionary of
    arguments.

    Returns agent name, agent args, and parse errors"""
    try:
        extracted_agent_name = extract_between_patterns(
            agent_content, "<AGENT_NAME>", "</AGENT_NAME>"
        )
        agent_args_str = extract_between_patterns(
            agent_content, "<AGENT_ARGS>", "</AGENT_ARGS>"
        )
        if extracted_agent_name is None:
            return None, None, "No agent name found in agent call"
        agent_name = extracted_agent_name

        agent_cls = agent_registry.get(agent_name)
        if agent_cls is None:
            return (None, None, f"{agent_name} does not correspond to a registered agent")

        if agent_args_str is None:
            return (None, None, f"Could not extract agent arguments in {agent_name} agent call")

        agent_args_dict, parse_warnings = await agent_cls.args_str_to_dict(
            agent_args_str, arg_format
        )
        if agent_args_dict is None:
            err = "Could not parse agent args"
            if parse_warnings:
                err = f"Could not parse agent args: {parse_warnings}"
            logger.info(f"Agent paarse error: {err}")

            return None, None, err

        return agent_name, agent_args_dict, parse_warnings

    except Exception as e:
        return None, None, f"Error parsing agent: {e}"


async def await_agent_task(
    agent_task: asyncio.Task, validated_agent: AgentInterface,
) -> AgentResult:
    callgraph = await CallGraphManager.get_instance()

    try:
        agent_result = await agent_task
        metrics = agent_result.metrics
        await callgraph.complete_agent(
            node_id=validated_agent._id,
            result=agent_result.result,
            token_count=metrics.token_usage.total_tokens,
            num_cached_tokens=metrics.token_usage.cached_prompt_tokens,
            cost=metrics.cost,
            success=agent_result.status == AgentStatus.SUCCESS,
        )
    except asyncio.CancelledError:
        logger.info(f"Agent {validated_agent.AGENT_NAME} was cancelled.")
        await callgraph.fail_agent(
            validated_agent._id, f"Agent {validated_agent.AGENT_NAME} was cancelled."
        )

        agent_result = AgentResult(
            agent_name=validated_agent.AGENT_NAME,
            status=AgentStatus.CANCELLED,
            result=f"Agent was cancelled",
            metrics=validated_agent._metrics,
        )
    except Exception as e:
        await callgraph.fail_agent(validated_agent._id, str(e))

        agent_result = AgentResult(
            agent_name=validated_agent.AGENT_NAME,
            status=AgentStatus.ERROR,
            result=f"Agent failed: {str(e)}",
            metrics=validated_agent._metrics,
            errors=str(e),
        )

    return agent_result


async def publish_agent_call(
    agent_content: ToolCallContent,
    calling_agent: AgentInterface
):
    """Publish an agent call event"""
    event_bus = await EventBus.get_instance()

    if agent_content.call_type == FCI.CONSTRAINED:
        agent_call_content = json.dumps(agent_content.tool_args)
    else:
        agent_call_content = f"""
<AGENT_CALL>
<AGENT_NAME>{agent_content.tool_name}</AGENT_NAME>
<AGENT_ARGS>
{dumps(agent_content.tool_args, calling_agent.MODEL.arg_format, indent=2)}
</AGENT_ARGS>
{AGENT_STOP_TOKEN}
"""
    await event_bus.publish(
        Event(
            type=EventType.AGENT_CALL,
            content=agent_call_content,
            metadata=dict(
                call_id=agent_content.call_id,
                name=agent_content.tool_name,
                args=agent_content.tool_args,
                call_type=agent_content.call_type,
                thought_signature=getattr(agent_content, "thought_signature", None),
            ),
        ),
        calling_agent._id,
    )


async def handle_parse_errors(
    agent_content: ToolCallContent,
    calling_agent: AgentInterface,
):
    parse_errors = agent_content.parse_errors

    await publish_agent_call(agent_content, calling_agent)

    event_bus = await EventBus.get_instance()
    logger.error(f"Agent parse error: {parse_errors}")
    await event_bus.publish(
        Event(
            type=EventType.APPLICATION_ERROR,
            content=f"Agent parse error: {parse_errors}",
        ),
        calling_agent._id,
    )

    # Add a error agent result so that the agent doesn't hallucinate it
    error_result = AgentResult(
        agent_name=agent_content.tool_name,
        status=AgentStatus.ERROR,
        result=f"Agent parse error: {parse_errors}",
        metrics=AgentMetrics(
            start_time=datetime.now(), end_time=datetime.now()
        ),
    )
    await event_bus.publish(
        Event(
            type=EventType.AGENT_RESULT,
            content=str(error_result),
            metadata=dict(
                call_id=agent_content.call_id,
                name=agent_content.tool_name,
                args=agent_content.tool_args,
                call_type=agent_content.call_type,
                agent_result=error_result
            ),
        ),
        calling_agent._id,
    )
    return


async def execute_agent_call(
    validated_agent: AgentInterface,
    calling_agent: AgentInterface,
    agent_content: ToolCallContent | None = None,
) -> AgentResult:
    """
    Executes the agent, taking care of
    a) registering it in the callgraph
    b) emittting the agent call and result events
    c) wrapping it in a cancellable task
    """
    event_bus = await EventBus.get_instance()
    agent_call_start_time = datetime.now()

    if agent_content is None:
        agent_content = ToolCallContent(
            call_id="none", tool_name=validated_agent.AGENT_NAME,
            tool_args=validated_agent.model_dump(), call_type=FCI.CONSTRAINED,
        )

    if validated_agent.AGENT_NAME not in calling_agent._available_agent_names:
        raise ValueError(
            f"{agent_content.tool_name} is not a member of the available agent set: {calling_agent._available_agents}"
        )

    try:
        await publish_agent_call(agent_content, calling_agent)

        # Register the agent with the callgraph
        callgraph = await CallGraphManager.get_instance()
        await callgraph.start_agent(
            agent_name=validated_agent.AGENT_NAME,
            node_id=validated_agent._id,
            args=validated_agent.model_dump(),
        )

        # Set off the agent task, and register it with the callgraph
        agent_task = asyncio.create_task(validated_agent.execute())
        await callgraph.register_agent_task(validated_agent._id, agent_task)

        agent_result = await await_agent_task(agent_task, validated_agent)

        await event_bus.publish(
            Event(
                type=EventType.AGENT_RESULT,
                content=str(agent_result),
                metadata=dict(
                    call_id=agent_content.call_id,
                    name=agent_content.tool_name,
                    args=agent_content.tool_args,
                    call_type=agent_content.call_type,
                    agent_result=agent_result
                ),
            ),
            calling_agent._id,
        )

        return agent_result
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        await event_bus.publish(
            Event(
                type=EventType.APPLICATION_ERROR,
                content=f"Agent runtime error: {e}",
            ),
            calling_agent._id,
        )

        # Add a error agent result so that the agent doesn't hallucinate it
        error_result = AgentResult(
            agent_name=(
                validated_agent.AGENT_NAME if validated_agent else "undefined"
            ),
            status=AgentStatus.ERROR,
            result=f"Agent runtime error: {e}",
            metrics=AgentMetrics(
                start_time=agent_call_start_time, end_time=datetime.now()
            ),
        )
        await event_bus.publish(
            Event(
                type=EventType.AGENT_RESULT,
                content=str(error_result),
                metadata=dict(
                    call_id=agent_content.call_id,
                    name=agent_content.tool_name,
                    args=agent_content.tool_args,
                    call_type=agent_content.call_type,
                    agent_result=error_result
                ),
            ),
            calling_agent._id,
        )
        return error_result
