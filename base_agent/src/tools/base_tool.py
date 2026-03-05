# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import time
import logging

from typing import ClassVar
from pydantic import TypeAdapter, PrivateAttr

from ..utils.parsing import extract_between_patterns
from ..utils.stop_tokens import TOOL_STOP_TOKEN
from ..events import EventBus
from ..schemas import get_schema_representation, args_str_to_dict, dumps
from ..types.tool_types import ToolInterface, ToolResult
from ..types.event_types import EventType, Event
from ..types.llm_types import FCI, ToolCallContent
from ..types.agent_types import AgentInterface
from ..types.common import ArgFormat

"""
TODO: ensure ordering of fields in native tool calling:

from pydantic import BaseModel, create_model
from pydantic.json_schema import GenerateJsonSchema
class OrderedGenerateJsonSchema(GenerateJsonSchema):
    def generate(self, schema, mode="validation"):
        schema_dict = super().generate(schema, mode=mode)
        # Automatically add propertyOrder based on field declaration order
        if hasattr(schema, "__fields__") and schema_dict.get("properties"):
            field_names = list(schema.__fields__.keys())
            schema_dict["propertyOrder"] = field_names
        return schema_dict

# Set this as the default schema generator
BaseModel.model_json_schema = classmethod(lambda cls, **kwargs: cls.__pydantic_core_schema__.schema_generator.generate(cls, **kwargs))
BaseModel.__pydantic_core_schema__.schema_generator = OrderedGenerateJsonSchema()

# Or use a decorator to apply it to specific models
def ordered_schema(cls):
    '''Decorator to automatically add property ordering to a Pydantic model's JSON schema'''
    original_schema = cls.model_json_schema
    def schema_with_order(*args, **kwargs):
        schema = original_schema(*args, **kwargs)
        # Get field names in order of declaration
        field_names = list(cls.model_fields.keys())
        schema["propertyOrder"] = field_names
        return schema
    cls.model_json_schema = classmethod(schema_with_order)
    return cls

# Usage:
@ordered_schema
class VerifyEditInput(BaseModel):
    reasoning: str
    is_complete: bool
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_tool_instructions(arg_format: ArgFormat = ArgFormat.XML) -> str:
    xml_instr = f"""When you need to use a tool, format your request as follows, with the tool arguments all inside the TOOL_ARGS tag, and the case-sensitive argument names within their own tags:"""
    json_instr = f"""When you need to use a tool, format your request using a valid JSON string inside this XML like structure:"""
    example_args = {"arg1": "value1", "arg2": "value2"}
    tool_instrs = f"""{json_instr if arg_format == ArgFormat.JSON else xml_instr}

<TOOL_CALL>
<TOOL_NAME>tool_name</TOOL_NAME>
<TOOL_ARGS>
{dumps(example_args, arg_format, indent=2)}
</TOOL_ARGS>
{TOOL_STOP_TOKEN}

Do not wrap the tool call in backticks like a code block. Simply start generating the <TOOL_CALL> block directly.

Note that you can reason and talk around the tool calls. For example your thought process might look like:
[... think about the problem...]
[tool call 1]
[... think about the result...]
[tool call 2]
[tool call 3]
[... think about all the results...]
"""

    return tool_instrs


async def parse_tool_content(
    tool_content: str, arg_format: ArgFormat
) -> tuple[str | None, dict | None, str | None]:
    """Returns tool name, tool args, and parse errors"""
    tool_name = "undefined"
    try:
        extracted_tool_name = extract_between_patterns(
            tool_content, "<TOOL_NAME>", "</TOOL_NAME>"
        )
        tool_args_str = extract_between_patterns(
            tool_content, "<TOOL_ARGS>", "</TOOL_ARGS>"
        )
        if extracted_tool_name is None:
            return None, None, "No tool name found in tool call"
        tool_name = extracted_tool_name

        tool_cls = tool_registry.get(tool_name)
        if tool_cls is None:
            return (None, None, f"{tool_name} does not correspond to a registered tool")

        if tool_args_str is None:
            return (
                None,
                None,
                f"Could not extract tool arguments in {tool_name} tool call",
            )

        tool_args_dict, parse_warnings = await tool_cls.args_str_to_dict(
            tool_args_str, arg_format
        )
        if tool_args_dict is None:
            err = "Could not parse tool args."
            if parse_warnings:
                err = f"Could not parse args: {parse_warnings}"
            logger.info(f"Tool parse error: {err}")

            return None, None, err

        return tool_name, tool_args_dict, parse_warnings

    except Exception as e:
        return None, None, f"Error parsing tool arguments: {e}"


# Create an empty registry dictionary.
# tool_registry: dict[str, type["BaseTool"]] = {}
tool_registry: dict[str, type[ToolInterface]] = {}


class BaseTool(ToolInterface):
    """Abstract base class for all tools"""

    # Class variables for tool metadata
    TOOL_NAME: ClassVar[str]
    TOOL_DESCRIPTION: ClassVar[str]

    # 'Ephemeral' tools are short-lived tools, registered and removed within an
    # agent's execution, which get their instructions injected directly into
    # the prefill and not the system message. A reasoning structure step is an
    # example.
    EPHEMERAL: ClassVar[bool] = False

    _calling_agent: AgentInterface = PrivateAttr()

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(**data)
        self._calling_agent = calling_agent

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip registering the BaseTool class itself.
        if cls.__name__ != "BaseTool":
            tool_registry[cls.TOOL_NAME] = cls

    # run, generate_examples are still abstract...

    @classmethod
    async def args_str_to_dict(
        cls, args_str: str, arg_format: ArgFormat = ArgFormat.XML
    ) -> tuple[dict | None, str | None]:
        args_dict, parse_warnings = await args_str_to_dict(
            args_str, guide_obj=cls, arg_format=arg_format, root_tag="TOOL_ARGS"
        )
        return args_dict, parse_warnings

    @classmethod
    def to_prompt_format(cls, arg_format: ArgFormat = ArgFormat.XML) -> str:
        """Convert the tool definition to XML format for the unconstrained tool use prompt"""
        examples = cls.generate_examples()

        examples_str = []
        for tool_instance, expected_output in examples:
            # instance_data = dict(
            #     tool_name=cls.TOOL_NAME, args=tool_instance.model_dump()
            # )
            # {dumps(instance_data, TOOL_ARGS, indent=2)}
            examples_str.append(
                f"""<EXAMPLE>
<TOOL_CALL>
<TOOL_NAME>{cls.TOOL_NAME}</TOOL_NAME>
<TOOL_ARGS>
{dumps(tool_instance.model_dump(), arg_format, indent=2)}
</TOOL_ARGS>
{TOOL_STOP_TOKEN}
{expected_output}
</EXAMPLE>"""
            )
        examples_joined = "\n\n".join(examples_str)

        return f"""\n## `{cls.TOOL_NAME}` Tool Documentation

{cls.TOOL_DESCRIPTION}

<TOOL_NAME>
{cls.TOOL_NAME}
</TOOL_NAME>
<TOOL_ARGS_SCHEMA>
{get_schema_representation(cls, arg_format=arg_format, root_tag="TOOL_ARGS")}
</TOOL_ARGS_SCHEMA>
<EXAMPLES>
{examples_joined}
</EXAMPLES>

This concludes the {cls.TOOL_NAME} tool documentation.
"""

    @classmethod
    def to_plain_prompt_format(cls, arg_format: ArgFormat = ArgFormat.XML) -> str:
        """Convert the tool definition to a formatted string for the constrained tool use prompt"""
        examples = cls.generate_examples()

        examples_str = []
        for i, (tool_instance, expected_output) in enumerate(examples):
            # instance_data = dict(
            #     tool_name=cls.TOOL_NAME, args=tool_instance.model_dump()
            # )
            # {dumps(instance_data, TOOL_ARGS, indent=2)}
            examples_str.append(
                f"""### `{cls.TOOL_NAME}` parameter example {i}

When the arguments are set to:
{dumps(tool_instance.model_dump(), arg_format, indent=2)}

The output might look like:
{expected_output}
"""
            )
        examples_joined = "\n\n".join(examples_str)

        return f"""\n## `{cls.TOOL_NAME}` Tool Documentation

{cls.TOOL_DESCRIPTION}

The tool's arguments are as follows:

{get_schema_representation(cls, arg_format=arg_format, root_tag="TOOL_ARGS")}

Here are some examples of how the arguments might be set and the associated results:

{examples_joined}

This concludes the {cls.TOOL_NAME} tool documentation.
"""


async def handle_tool_call(
    tool_content: ToolCallContent, calling_agent: AgentInterface
):
    """Handle a tool call, and adds the TOOL_CALL and TOOL_RESULT events
    to the event bus"""
    event_bus = await EventBus.get_instance()
    validated_tool = None

    if tool_content.call_type == FCI.CONSTRAINED:
        tool_call_content = json.dumps(tool_content.tool_args)
    else:
        tool_call_content = f"""
<TOOL_CALL>
<TOOL_NAME>{tool_content.tool_name}</TOOL_NAME>
<TOOL_ARGS>
{dumps(tool_content.tool_args, calling_agent.MODEL.arg_format, indent=2)}
</TOOL_ARGS>
{TOOL_STOP_TOKEN}
"""

    try:
        tool_cls = tool_registry.get(tool_content.tool_name)
        parse_errors = tool_content.parse_errors

        if parse_errors:
            logger.error(f"Tool parse error: {parse_errors}")
            await event_bus.publish(
                Event(
                    type=EventType.APPLICATION_ERROR,
                    content=f"This tool call could not be parsed: {parse_errors}",
                ),
                calling_agent._id,
            )

            # Make sure to publish the event to correctly re-construct the
            # prefill in unconstrained mode, or in order to publish the
            # tool result in constrained mode.
            await event_bus.publish(
                Event(
                    type=EventType.TOOL_CALL,
                    content=tool_call_content,
                    metadata=dict(
                        call_id=tool_content.call_id,
                        name=tool_content.tool_name,
                        args=tool_content.tool_args,
                        call_type=tool_content.call_type,
                        thought_signature=getattr(tool_content, "thought_signature", None),
                    ),
                ),
                calling_agent._id,
            )

            # Add a error tool result so that the agent doesn't hallucinate it
            error_result = ToolResult(
                tool_name=tool_content.tool_name,
                success=False,
                output=f"This tool call could not be parsed: {parse_errors}",
            )
            await event_bus.publish(
                Event(
                    type=EventType.TOOL_RESULT,
                    content=str(error_result),
                    metadata=dict(
                        call_id=tool_content.call_id,
                        name=tool_content.tool_name,
                        args=tool_content.tool_args,
                        call_type=tool_content.call_type,
                        tool_result=error_result,
                    ),
                ),
                calling_agent._id,
            )
            return

        validated_tool = TypeAdapter(tool_cls).validate_python(
            tool_content.tool_args | {"calling_agent": calling_agent}
        )

        await event_bus.publish(
            Event(
                type=EventType.TOOL_CALL,
                content=tool_call_content,
                metadata=dict(
                    call_id=tool_content.call_id,
                    name=validated_tool.TOOL_NAME,
                    args=validated_tool.model_dump(),
                    call_type=tool_content.call_type,
                    thought_signature=getattr(tool_content, "thought_signature", None),
                ),
            ),
            calling_agent._id,
        )

        start_time = time.time()
        tool_result = await validated_tool.run()
        tool_result.duration = time.time() - start_time

        if parse_errors:
            warning_msg = f"\nThe tool call was successful, but raised the following warnings during parsing and fixing: {parse_errors}\nNext time you call the tool, try to avoid these errors."
            if tool_result.warnings:
                tool_result.warnings += warning_msg
            else:
                tool_result.warnings = warning_msg

        await event_bus.publish(
            Event(
                type=EventType.TOOL_RESULT,
                content=str(tool_result),
                metadata=dict(
                    call_id=tool_content.call_id,
                    name=tool_content.tool_name,
                    args=validated_tool.model_dump(),
                    call_type=tool_content.call_type,
                    tool_result=tool_result,
                ),
            ),
            calling_agent._id,
        )

    except Exception as e:
        logger.error(f"Error during tool execution: {str(e)}")
        await event_bus.publish(
            Event(
                type=EventType.APPLICATION_ERROR,
                content=f"Tool runtime error: {str(e)}",
            ),
            calling_agent._id,
        )

        # Cannot publish this without first publishing the tool call in
        # constrained mode
        # error_result = ToolResult(
        #     tool_name=(
        #         validated_tool.TOOL_NAME if validated_tool else "unknown_tool"
        #     ),
        #     success=False,
        #     output=f"Tool runtime error: {str(e)}",
        # )
        # await event_bus.publish(
        #     Event(
        #         type=EventType.TOOL_RESULT,
        #         content=str(error_result),
        #         metadata=dict(
        #             call_id=tool_content.call_id,
        #             name=tool_content.tool_name,
        #             args=tool_content.tool_args,
        #             call_type=tool_content.call_type,
        #             tool_result=error_result,
        #         ),
        #     ),
        #     calling_agent._id,
        # )
