import json
import logging
from enum import Enum
from inspect import signature
from secrets import token_hex
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    Union,
    cast,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from agno.agent import Agent
from agno.models.base import Model
from agno.team import Team
from agno.tools.function import Function, FunctionCall
from agno.tools.toolkit import Toolkit
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_AGNO_PARENT_NODE_CONTEXT_KEY = context_api.create_key("agno_parent_node_id")


def _flatten(mapping: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    if not mapping:
        return
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


def _get_input_value(method: Callable[..., Any], instance: Any, *args: Any, **kwargs: Any) -> str:
    arguments = _bind_arguments(method, instance, *args, **kwargs)
    return _extract_agent_input(arguments)


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    try:
        method_signature = signature(method)
        bound_args = method_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arguments = bound_args.arguments
        # Remove 'self' from arguments as it's the instance
        arguments.pop("self", None)
        arguments = OrderedDict(
            {key: value for key, value in arguments.items() if value is not None and value != {}}
        )
        return dict(arguments)
    except (TypeError, ValueError):
        # Try alternate approach if agno is using kwargs only
        if kwargs:
            # Just return the kwargs directly, they're already the arguments we want
            return dict(kwargs)
        # If binding fails, return empty dict to avoid breaking the wrapper
        return {}


def _extract_agent_input(arguments: Mapping[str, Any]) -> str:
    """Extract input for Agent/Team spans."""
    import dataclasses

    # Filter out internal parameters
    excluded_params = {"self", "cls", "run_response", "run_messages"}
    result = {}

    for key, value in arguments.items():
        if key in excluded_params:
            continue

        if key == "session" and value is not None:
            # AgentSession or TeamSession - they're dataclasses with to_dict method
            if hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                # Only call asdict if it's an instance, not a class
                result[key] = dataclasses.asdict(value)
            else:
                # This shouldn't happen for session objects
                result[key] = str(value)
        else:
            # Other parameters pass through as-is
            result[key] = value

    return safe_json_dumps(result)


def _generate_node_id() -> str:
    return token_hex(8)  # Generates 16 hex characters (8 bytes)


def _extract_agent_output(response: Any) -> str:
    """Extract output for Agent/Team spans - expects ModelResponse."""
    import dataclasses

    if response is None:
        return "{}"

    # ModelResponse is a dataclass, use dataclasses.asdict
    if dataclasses.is_dataclass(response) and not isinstance(response, type):
        # Convert to dict and remove None values
        response_dict = dataclasses.asdict(response)
        return safe_json_dumps(_remove_none_values(response_dict))

    return safe_json_dumps(response)


def _remove_none_values(obj: Any) -> Any:
    """Recursively remove None values from dictionaries."""
    if isinstance(obj, dict):
        return {k: _remove_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [_remove_none_values(item) for item in obj]
    else:
        return obj


def _serialize_dataclass_with_pydantic_fields(obj: Any) -> Any:
    """Serialize dataclass fields, handling nested Pydantic models properly.

    When a dataclass (like ModelResponse) contains Pydantic models (like Message),
    dataclasses.asdict() converts them to dicts with ALL fields including None values.
    This function properly serializes them using their model_dump method.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if v is None:
                continue  # Skip None values

            # Check if this looks like a serialized Pydantic model
            # (has many fields that Pydantic models typically have)
            if isinstance(v, dict) and "metrics" in v and "created_at" in v:
                # This is likely a Message that was converted to dict by dataclasses.asdict
                # We can't use model_dump here since it's already a dict, so just clean it
                result[k] = _remove_none_values(v)
            elif isinstance(v, list):
                # Handle lists of potentially Pydantic models
                result[k] = [_serialize_dataclass_with_pydantic_fields(item) for item in v]
            elif isinstance(v, dict):
                # Recursively handle nested dicts
                result[k] = _serialize_dataclass_with_pydantic_fields(v)
            else:
                result[k] = v
        return result
    elif isinstance(obj, list):
        return [_serialize_dataclass_with_pydantic_fields(item) for item in obj]
    else:
        return obj


def _run_arguments(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")

    # For Team._run, session_id is in the session object
    session = arguments.get("session")
    if session:
        if hasattr(session, "session_id"):
            session_id = session.session_id
        if hasattr(session, "user_id") and not user_id:
            user_id = session.user_id

    if session_id:
        yield SESSION_ID, session_id

    if user_id:
        yield USER_ID, user_id


def _agent_run_attributes(
    agent: Union[Agent, Team], key_suffix: str = ""
) -> Iterator[Tuple[str, AttributeValue]]:
    # Get parent from execution context instead of structural parent
    context_parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

    if isinstance(agent, Team):
        # Set graph attributes for team - these are the main attributes for this span
        if agent.name:
            yield GRAPH_NODE_NAME, agent.name

        # Use context parent instead of structural parent
        if context_parent_id:
            yield GRAPH_NODE_PARENT_ID, cast(str, context_parent_id)

        # Set team-specific attributes
        yield f"agno{key_suffix}.team", agent.name or ""

        # Add member information as nested attributes (not graph attributes)
        for member in agent.members:
            if member.name:
                yield f"agno.{member.name}.agent", member.name
            if member.tools:
                tool_names = []
                for tool in member.tools:
                    if isinstance(tool, Function):
                        tool_names.append(tool.name)
                    elif isinstance(tool, Toolkit):
                        tool_names.extend(sorted(tool.functions.keys()))  # Sort toolkit functions
                    elif callable(tool):
                        tool_names.append(tool.__name__)
                    else:
                        tool_names.append(str(tool))
                # Sort tool names for predictable ordering
                yield f"agno.{member.name}.tools", sorted(tool_names)

    elif isinstance(agent, Agent):
        # Set graph attributes for agent
        if agent.name:
            yield GRAPH_NODE_NAME, agent.name

        # Use context parent instead of structural parent
        if context_parent_id:
            yield GRAPH_NODE_PARENT_ID, cast(str, context_parent_id)

        # Set agent-specific attributes
        if agent.name:
            yield f"agno{key_suffix}.agent", agent.name or ""

        if agent.knowledge:
            yield f"agno{key_suffix}.knowledge", agent.knowledge.__class__.__name__

        # Always set agno.tools, even if empty (agno 2.0 always has tools attribute)
        tool_names = []
        if agent.tools:
            for tool in agent.tools:
                if isinstance(tool, Function):
                    tool_names.append(tool.name)
                elif isinstance(tool, Toolkit):
                    tool_names.extend(sorted(tool.functions.keys()))  # Sort toolkit functions
                elif callable(tool):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
        # Sort tool names for predictable ordering
        yield f"agno{key_suffix}.tools", tuple(sorted(tool_names))


def _setup_team_context(agent: Union[Agent, Team], node_id: str) -> Optional[Any]:
    if isinstance(agent, Team):
        team_ctx = context_api.set_value(_AGNO_PARENT_NODE_CONTEXT_KEY, node_id)
        return context_api.attach(team_ctx)
    return None


class _RunWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    """
    We need to keep track of parent/child relationships for agent logging. We do this by:
    1. Each run() method generates a unique node_id and sets it directly as GRAPH_NODE_ID in span
    attributes
    2. Team.run() sets _AGNO_PARENT_NODE_CONTEXT_KEY for child agents
    3. Agent.run() inherits _AGNO_PARENT_NODE_CONTEXT_KEY from team context for parent relationships
    4. _agent_run_attributes() uses _AGNO_PARENT_NODE_CONTEXT_KEY to set GRAPH_NODE_PARENT_ID
    5. This ensures correct parent-child relationships with unique node IDs for each execution
    """

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        arguments = _bind_arguments(wrapped, instance, *args, **kwargs)

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            instance,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                result = wrapped(*args, **kwargs)
                span.set_status(trace_api.StatusCode.OK)

                # For Team._run, the response is in the arguments, not the return value
                if result is not None:
                    span.set_attribute(OUTPUT_VALUE, _extract_agent_output(result))
                    span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                elif "run_response" in arguments:
                    # Team._run passes run_response as a parameter and modifies it
                    run_response_arg = arguments["run_response"]
                    if run_response_arg is not None:
                        span.set_attribute(OUTPUT_VALUE, _extract_agent_output(run_response_arg))
                        span.set_attribute(OUTPUT_MIME_TYPE, JSON)

                return result

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        arguments = _bind_arguments(wrapped, instance, *args, **kwargs)

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            instance,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                yield from wrapped(*args, **kwargs)
                run_response = agent.run_response
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, _extract_agent_output(run_response))
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            response = await wrapped(*args, **kwargs)
            return response

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.arun"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        arguments = _bind_arguments(wrapped, instance, *args, **kwargs)

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            instance,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                result = await wrapped(*args, **kwargs)
                span.set_status(trace_api.StatusCode.OK)

                # For Team._arun, the response is in the arguments, not the return value
                if result is not None:
                    span.set_attribute(OUTPUT_VALUE, _extract_agent_output(result))
                    span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                elif "run_response" in arguments:
                    # Team._arun passes run_response as a parameter and modifies it
                    run_response_arg = arguments["run_response"]
                    if run_response_arg is not None:
                        span.set_attribute(OUTPUT_VALUE, _extract_agent_output(run_response_arg))
                        span.set_attribute(OUTPUT_MIME_TYPE, JSON)

                return result
            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)

    async def arun_stream(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for response in await wrapped(*args, **kwargs):
                yield response

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.arun_stream"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        arguments = _bind_arguments(wrapped, instance, *args, **kwargs)

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            instance,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                async for response in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                    yield response
                run_response = agent.run_response
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, _extract_agent_output(run_response))
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)


def _llm_input_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    def process_message(idx: int, role: str, content: str) -> Iterator[Tuple[str, Any]]:
        yield f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_ROLE}", role
        yield f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_CONTENT}", content

    messages = arguments.get("messages", [])
    for i, message in enumerate(messages):
        role, content = message.role, message.get_content_string()
        if content:
            yield from process_message(i, role, content)

    tools = arguments.get("tools", [])
    for tool_index, tool in enumerate(tools):
        yield f"{LLM_TOOLS}.{tool_index}.{TOOL_JSON_SCHEMA}", safe_json_dumps(tool)


def _llm_invocation_parameters(
    model: Model, arguments: Optional[Mapping[str, Any]] = None
) -> Iterator[Tuple[str, Any]]:
    request_kwargs = {}
    # TODO (v2.0.0): with the cleanup of the agno.models.base.Model class we will
    # handle these attributes in a more consistent way.
    if getattr(model, "request_kwargs", None):
        request_kwargs = model.request_kwargs  # type: ignore[attr-defined]
    if getattr(model, "request_params", None):
        request_kwargs = model.request_params  # type: ignore[attr-defined]
    if getattr(model, "get_request_kwargs", None):
        request_kwargs = model.get_request_kwargs()  # type: ignore[attr-defined]
    if getattr(model, "get_request_params", None):
        # Special handling for OpenAIResponses model
        if model.__class__.__name__ == "OpenAIResponses" and arguments:
            messages = arguments.get("messages", [])
            request_kwargs = model.get_request_params(messages=messages)  # type: ignore[attr-defined]
        else:
            request_kwargs = model.get_request_params()  # type: ignore[attr-defined]

    if request_kwargs:
        filtered_kwargs = _filter_sensitive_params(request_kwargs)
        yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(filtered_kwargs)


def _filter_sensitive_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out sensitive parameters from model request parameters."""
    sensitive_keys = frozenset(
        [
            "api_key",
            "api_base",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_access_key",
            "aws_secret_key",
            "azure_endpoint",
            "azure_deployment",
            "azure_ad_token",
            "azure_ad_token_provider",
        ]
    )

    return {
        key: "[REDACTED]"
        if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys)
        else value
        for key, value in params.items()
    }


def _input_value_and_mime_type(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _output_value_and_mime_type(output: str) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_MIME_TYPE, JSON
    yield OUTPUT_VALUE, output


def _parse_model_output(output: Any) -> str:
    """Parse model output to JSON string.

    Handles:
    - Pydantic models (Message) with model_dump_json
    - Dataclasses (ModelResponse) with custom serialization
    - Dicts and other types
    """
    import dataclasses

    # Check for Pydantic model first (like Message)
    if hasattr(output, "model_dump_json"):
        # Use Pydantic's built-in JSON serialization with exclude_none
        return output.model_dump_json(exclude_none=True)  # type: ignore[no-any-return]

    # Then check for dataclass (like ModelResponse)
    elif dataclasses.is_dataclass(output) and not isinstance(output, type):
        # Convert to dict
        output_dict = dataclasses.asdict(output)

        # Special handling for nested Pydantic models in dataclass fields
        cleaned_dict = _serialize_dataclass_with_pydantic_fields(output_dict)
        return json.dumps(cleaned_dict)

    elif isinstance(output, dict):
        return json.dumps(_remove_none_values(output))

    else:
        return str(output)


class _ModelWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, instance, *args, **kwargs)

        model = instance
        span_name = "ChatCompletion"

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(_llm_invocation_parameters(model, arguments)),
                        **dict(_llm_input_messages(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            response = wrapped(*args, **kwargs)
            output_message = _parse_model_output(response)

            span.set_attributes(dict(_output_value_and_mime_type(output_message)))
            return response

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, instance, *args, **kwargs)

        model = instance
        span_name = "ChatCompletion"

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(_llm_invocation_parameters(model)),
                        **dict(_llm_input_messages(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            responses = []
            for chunk in wrapped(*args, **kwargs):
                responses.append(chunk)
                yield chunk
            output_message = json.dumps([_parse_model_output(response) for response in responses])
            span.set_attributes(dict(_output_value_and_mime_type(output_message)))

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, instance, *args, **kwargs)

        model = instance
        span_name = "ChatCompletion"

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(_llm_invocation_parameters(model)),
                        **dict(_llm_input_messages(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            response = await wrapped(*args, **kwargs)
            output_message = _parse_model_output(response)

            span.set_attributes(dict(_output_value_and_mime_type(output_message)))
            return response

    async def arun_stream(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for response in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                yield response
            return

        arguments = _bind_arguments(wrapped, instance, *args, **kwargs)

        model = instance
        span_name = "ChatCompletion"

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(_llm_invocation_parameters(model)),
                        **dict(_llm_input_messages(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            responses = []
            async for chunk in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                responses.append(chunk)
                yield chunk
            output_message = json.dumps([_parse_model_output(response) for response in responses])
            span.set_attributes(dict(_output_value_and_mime_type(output_message)))


def _function_call_attributes(function_call: FunctionCall) -> Iterator[Tuple[str, Any]]:
    function = function_call.function
    function_name = function.name
    function_arguments = function_call.arguments

    yield TOOL_NAME, function_name

    if function_description := getattr(function, "description", None):
        yield TOOL_DESCRIPTION, function_description
    yield TOOL_PARAMETERS, safe_json_dumps(function_arguments)


def _input_value_and_mime_type_for_tool_span(
    arguments: Mapping[str, Any],
) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _output_value_and_mime_type_for_tool_span(result: Any) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_VALUE, str(result)
    yield OUTPUT_MIME_TYPE, TEXT


class _FunctionCallWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        function_call = instance
        function = function_call.function
        function_name = function.name
        function_arguments = function_call.arguments

        span_name = f"{function_name}"

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: TOOL,
                        **dict(_input_value_and_mime_type_for_tool_span(function_arguments)),
                        **dict(_function_call_attributes(function_call)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            success = wrapped(*args, **kwargs)  # Returns bool in agno 1.5.2

            if success:
                function_result = function_call.result
                span.set_status(trace_api.StatusCode.OK)
                span.set_attributes(
                    dict(
                        _output_value_and_mime_type_for_tool_span(
                            result=function_result,
                        )
                    )
                )
            else:
                function_error_message = function_call.error
                span.set_status(trace_api.StatusCode.ERROR, function_error_message)
                span.set_attribute(OUTPUT_VALUE, function_error_message)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

        return success

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        function_call = instance
        function = function_call.function
        function_name = function.name
        function_arguments = function_call.arguments

        # Keep the tool name without prefix for consistency
        span_name = f"{function_name}"

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: TOOL,
                        **dict(_input_value_and_mime_type_for_tool_span(function_arguments)),
                        **dict(_function_call_attributes(function_call)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            success = await wrapped(*args, **kwargs)  # Returns bool in agno 1.5.2

            if success:
                function_result = function_call.result
                span.set_status(trace_api.StatusCode.OK)
                span.set_attributes(
                    dict(
                        _output_value_and_mime_type_for_tool_span(
                            result=function_result,
                        )
                    )
                )
            else:
                function_error_message = function_call.error
                span.set_status(trace_api.StatusCode.ERROR, function_error_message)
                span.set_attribute(OUTPUT_VALUE, function_error_message)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

        return success


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
SESSION_ID = SpanAttributes.SESSION_ID
LLM_TOOLS = SpanAttributes.LLM_TOOLS
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_FUNCTION_CALL = SpanAttributes.LLM_FUNCTION_CALL
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS
USER_ID = SpanAttributes.USER_ID
GRAPH_NODE_ID = SpanAttributes.GRAPH_NODE_ID
GRAPH_NODE_NAME = SpanAttributes.GRAPH_NODE_NAME
GRAPH_NODE_PARENT_ID = SpanAttributes.GRAPH_NODE_PARENT_ID

# message attributes
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS

# mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# span kinds
AGENT = OpenInferenceSpanKindValues.AGENT.value
LLM = OpenInferenceSpanKindValues.LLM.value
TOOL = OpenInferenceSpanKindValues.TOOL.value

# tool attributes
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

# tool call attributes
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
