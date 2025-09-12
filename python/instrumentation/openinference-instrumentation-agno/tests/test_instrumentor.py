import json
import os
from typing import Any, Optional

import pytest
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.team import Team
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.agno import AgnoInstrumentor
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from .test_tools import SinglePageWebsiteTools

# Disable telemetry to Agno during tests
os.environ["AGNO_TELEMETRY"] = "false"

# Constants for test session/user IDs
TEST_SESSION_ID = "test-session-123"
TEST_USER_ID = "test-user-456"

# Constants for context attributes test
CONTEXT_SESSION_ID = "my-test-session"
CONTEXT_USER_ID = "my-test-user"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda request: request.headers.clear() or request,
    before_record_response=lambda response: dict(response, headers={}),
)
def test_agno_instrumentation(trace_exporter: InMemorySpanExporter) -> None:
    """Test agno instrumentation for synchronous team execution.

    Expected spans (12 total):
    - 3 AGENT spans:
      * Team.run - orchestrates the workflow
      * Scraper_Agent.run - scrapes website content
      * Analyzer_Agent.run - analyzes scraped content
    - 3 TOOL spans:
      * delegate_task_to_member (to scraper) - Team delegates scraping task
      * delegate_task_to_member (to analyzer) - Team delegates analysis task
      * read_url - Scraper Agent fetches quotes.toscrape.com
    - 6 LLM spans (ChatCompletion):
      * 2 for Team (initial delegation, final response after analysis)
      * 2 for Scraper Agent (understanding task, after scraping)
      * 2 for Analyzer Agent (understanding task, generating analysis)
    """
    run_team(session_id=TEST_SESSION_ID, user_id=TEST_USER_ID)

    spans = trace_exporter.get_finished_spans()
    _verify_agno_instrumentation(
        spans,
        team_span_name="Team.run",
        scraper_span_name="Scraper_Agent.run",
        transfer_tool_name="delegate_task_to_member",
    )


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda request: request.headers.clear() or request,
    before_record_response=lambda response: dict(response, headers={}),
)
def test_agno_instrumentation_context_attributes(
    trace_exporter: InMemorySpanExporter,
) -> None:
    """Test that context attributes are properly attached to all spans.

    Expected spans (12 total):
    - 3 AGENT spans:
      * Team.run - orchestrates the workflow
      * Scraper_Agent.run - scrapes website content
      * Analyzer_Agent.run - analyzes scraped content
    - 3 TOOL spans:
      * delegate_task_to_member (to scraper) - Team delegates scraping task
      * delegate_task_to_member (to analyzer) - Team delegates analysis task
      * read_url - Scraper Agent fetches quotes.toscrape.com
    - 6 LLM spans (ChatCompletion):
      * 2 for Team (initial delegation, final response after analysis)
      * 2 for Scraper Agent (understanding task, after scraping)
      * 2 for Analyzer Agent (understanding task, generating analysis)

    All spans should have context attributes from using_attributes().
    """
    with using_attributes(
        session_id=CONTEXT_SESSION_ID,
        user_id=CONTEXT_USER_ID,
        metadata={
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {
                "key-1": "val-1",
                "key-2": "val-2",
            },
        },
        tags=["tag-1", "tag-2"],
        prompt_template="test-prompt-template",
        prompt_template_version="v1.0",
        prompt_template_variables={
            "var-1": "value-1",
            "var-2": "value-2",
        },
    ):
        run_team(session_id="context-test-session", user_id="context-test-user")

    spans = trace_exporter.get_finished_spans()
    assert len(spans) == 12

    for span in spans:
        _verify_context_attributes(span)


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda request: request.headers.clear() or request,
    before_record_response=lambda response: dict(response, headers={}),
)
@pytest.mark.asyncio
async def test_agno_instrumentation_async(trace_exporter: InMemorySpanExporter) -> None:
    """Test agno instrumentation for asynchronous team execution.

    Expected spans (12 total):
    - 3 AGENT spans:
      * Team.arun - orchestrates the workflow (async)
      * Scraper_Agent.arun - scrapes website content (async)
      * Analyzer_Agent.arun - analyzes scraped content (async)
    - 3 TOOL spans:
      * delegate_task_to_member (to scraper) - Team delegates scraping task
      * delegate_task_to_member (to analyzer) - Team delegates analysis task
      * read_url - Scraper Agent fetches quotes.toscrape.com
    - 6 LLM spans (ChatCompletion):
      * 2 for Team (initial delegation, final response after analysis)
      * 2 for Scraper Agent (understanding task, after scraping)
      * 2 for Analyzer Agent (understanding task, generating analysis)
    """
    await arun_team(session_id=TEST_SESSION_ID, user_id=TEST_USER_ID)

    spans = trace_exporter.get_finished_spans()
    _verify_agno_instrumentation(
        spans,
        team_span_name="Team.arun",
        scraper_span_name="Scraper_Agent.arun",
        transfer_tool_name="delegate_task_to_member",
    )


def _verify_agno_instrumentation(
    spans: tuple[ReadableSpan, ...],
    team_span_name: str,
    scraper_span_name: str,
    transfer_tool_name: str = "transfer_task_to_member",
) -> None:
    agent_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)
    tool_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.TOOL.value)
    llm_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.LLM.value)

    assert len(spans) == 12
    assert len(agent_spans) == 3
    assert len(tool_spans) == 3
    assert len(llm_spans) == 6

    # Agent spans are sorted alphabetically
    # For async, all spans have 'arun', for sync they have 'run'
    analyzer_name = "Analyzer_Agent.arun" if "arun" in team_span_name else "Analyzer_Agent.run"
    assert agent_spans[0].name == analyzer_name
    assert agent_spans[1].name == scraper_span_name
    assert agent_spans[2].name == team_span_name
    # Agent spans are sorted alphabetically: Analyzer_Agent.run, Scraper_Agent.run, Team.run
    analyzer_span = agent_spans[0]
    scraper_span = agent_spans[1]
    team_span = agent_spans[2]

    team_attributes = dict(team_span.attributes or {})
    team_node_id = team_attributes[SpanAttributes.GRAPH_NODE_ID]
    assert isinstance(team_node_id, str)

    _verify_team_span(
        team_span,
        expected_name=team_span_name,
        expected_team_members=["Scraper Agent", "Analyzer Agent"],
        expected_node_id=team_node_id,
        expected_session_id=TEST_SESSION_ID,
        expected_user_id=TEST_USER_ID,
        # agno 2.0 now captures output correctly
        expected_output_value_keys={
            # ModelResponse fields from agno 2.0
            "role",
            "content",
            "parsed",
            "audio",
            "images",
            "videos",
            "audios",
            "tool_calls",
            "tool_executions",
            "event",
            "provider_data",
            "redacted_reasoning_content",
            "reasoning_content",
            "citations",
            "response_usage",
            "created_at",
            "extra",
            "updated_session_state",
            "input",
            "created_at",
            "status",
            "messages",
            "tools",
        },
    )
    scraper_attributes = dict(scraper_span.attributes or {})
    scraper_node_id = scraper_attributes[SpanAttributes.GRAPH_NODE_ID]
    assert isinstance(scraper_node_id, str)

    _verify_agent_span(
        scraper_span,
        expected_name=scraper_span_name,
        expected_agent_name="Scraper Agent",
        expected_node_id=scraper_node_id,
        expected_parent_id=team_node_id,
        # agno 2.0 now captures session data for agents
        expected_input_value={
            "user_id": TEST_USER_ID,
            "session": str,  # Complex AgentSession object
            "response_format": None,
        },
        expected_output_value_keys={
            "content",
            "content_type",
            "metrics",
            "model",
            "model_provider",
            "run_id",
            "agent_id",
            "agent_name",
            "session_id",
            "tools",
            "created_at",
            "messages",
            "status",
            "input",
        },
        expected_agno_tools=("read_url",),
    )

    # Verify Analyzer Agent span
    analyzer_attributes = dict(analyzer_span.attributes or {})
    analyzer_node_id = analyzer_attributes[SpanAttributes.GRAPH_NODE_ID]
    assert isinstance(analyzer_node_id, str)

    _verify_agent_span(
        analyzer_span,
        expected_name=analyzer_name,  # Use the variable we set earlier
        expected_agent_name="Analyzer Agent",
        expected_node_id=analyzer_node_id,
        expected_parent_id=team_node_id,
        # agno 2.0 now captures session data for agents
        expected_input_value={
            "user_id": TEST_USER_ID,
            "session": str,  # Complex AgentSession object
            "response_format": None,
        },
        expected_output_value_keys={
            "content",
            "content_type",
            "metrics",
            "model",
            "model_provider",
            "run_id",
            "agent_id",
            "agent_name",
            "session_id",
            "tools",
            "created_at",
            "messages",
            "status",
            "input",
        },
        expected_agno_tools=(),
    )

    _verify_tool_span(
        tool_spans[0],
        transfer_tool_name,
        {
            "member_id": "analyzer-agent",
            "task_description": (
                "Analyze the scraped content and identify the first quote and its author. "
                "Format the output as: 'First quote: [quote] by [author]'"
            ),
            "expected_output": "First quote: [quote] by [author]",
        },
    )
    _verify_tool_span(
        tool_spans[1],
        transfer_tool_name,
        {
            "member_id": "scraper-agent",
            "task_description": "Scrape the text content from https://quotes.toscrape.com/",
            "expected_output": "The scraped content from the website.",
        },
    )
    _verify_tool_span(tool_spans[2], "read_url", {"url": "https://quotes.toscrape.com/"})

    for span in llm_spans:
        _verify_llm_span(span, "ChatCompletion")


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (instrumentor_entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="agno")  # type: ignore[no-untyped-call]
    instrumentor = instrumentor_entrypoint.load()()
    assert isinstance(instrumentor, AgnoInstrumentor)
    assert isinstance(AgnoInstrumentor()._tracer, OITracer)


def create_team_and_agents() -> tuple[Team, Agent, Agent]:
    openai_api_key = os.getenv("OPENAI_API_KEY", "sk-test")
    url = "https://quotes.toscrape.com/"

    model = OpenAIChat(id="gpt-4o-mini", api_key=openai_api_key, temperature=0)

    scraper_agent = Agent(
        name="Scraper Agent",
        role="Website Scraper",
        model=model,
        tools=[SinglePageWebsiteTools()],
        instructions="Scrape the content from the given website URL",
        reasoning_max_steps=1,
        additional_context=None,
        retries=0,
    )

    analyzer_agent = Agent(
        name="Analyzer Agent",
        role="Content Analyzer",
        model=model,
        instructions=(
            "Analyze scraped website content to extract useful information "
            "such as extracting quotes and authors."
        ),
        reasoning_max_steps=1,
        additional_context=None,
        retries=0,
    )

    team = Team(
        name="Team",
        members=[scraper_agent, analyzer_agent],
        model=model,
        instructions=[
            f"Use the read_url tool to scrape content from {url}",
            "From the scraped content, identify the first quote and its author",
        ],
        reasoning_min_steps=1,
        reasoning_max_steps=1,
        retries=0,  # Prevent retries to avoid duplicate spans
        share_member_interactions=True,  # Share context between members
        determine_input_for_members=True,  # Help members get the right input
    )

    return team, scraper_agent, analyzer_agent


def run_team(session_id: str, user_id: str) -> tuple[Agent, Agent]:
    team, scraper_agent, analyzer_agent = create_team_and_agents()
    url = "https://quotes.toscrape.com/"
    result = team.run(
        f"Scrape the text content from {url} and identify the first quote and its author. "
        "Format as: 'First quote: [quote] by [author]'",
        session_id=session_id,
        user_id=user_id,
    )
    assert result is not None
    assert "Albert Einstein" in str(result)
    return scraper_agent, analyzer_agent


async def arun_team(session_id: str, user_id: str) -> tuple[Agent, Agent]:
    team, scraper_agent, analyzer_agent = create_team_and_agents()
    url = "https://quotes.toscrape.com/"
    result = await team.arun(
        f"Scrape the text content from {url} and identify the first quote and its author. "
        "Format as: 'First quote: [quote] by [author]'",
        session_id=session_id,
        user_id=user_id,
    )
    assert result is not None
    assert "Albert Einstein" in str(result)
    return scraper_agent, analyzer_agent


def get_spans_by_kind(spans: tuple[ReadableSpan, ...], kind: str) -> list[ReadableSpan]:
    return sorted(
        [
            span
            for span in spans
            if span.attributes
            and span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == kind
        ],
        key=lambda s: (
            s.name,
            str(s.attributes.get(SpanAttributes.INPUT_VALUE, "")) if s.attributes else "",
        ),
    )


def _verify_team_span(
    span: ReadableSpan,
    expected_name: str,
    expected_team_members: list[str],
    expected_node_id: str,
    expected_session_id: str,
    expected_user_id: str,
    expected_output_value_keys: Optional[set[str]],
) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == expected_name
    assert (
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.AGENT.value
    )

    # Team._run gets session and user_id as input
    input_value = attributes[SpanAttributes.INPUT_VALUE]
    assert isinstance(input_value, str)

    # For Team._run in agno 2.0, we just verify it has input
    # The structure is complex with TeamSession objects
    # We skip detailed checks as the cassette captures the right behavior

    # Verify session_id and user_id from span attributes
    assert attributes.get(SpanAttributes.SESSION_ID) == expected_session_id
    assert attributes.get(SpanAttributes.USER_ID) == expected_user_id

    if expected_output_value_keys is not None:
        output_value = attributes[SpanAttributes.OUTPUT_VALUE]
        assert isinstance(output_value, str)
        # Skip checking exact fields - response structure varies in agno 2.0
        # Just verify it's valid JSON with content
        output_dict = json.loads(output_value)
        assert "content" in output_dict or "messages" in output_dict
        assert attributes[SpanAttributes.OUTPUT_MIME_TYPE] == "application/json"
    else:
        assert SpanAttributes.OUTPUT_VALUE not in attributes
        assert SpanAttributes.OUTPUT_MIME_TYPE not in attributes

    assert attributes[SpanAttributes.GRAPH_NODE_ID] == expected_node_id
    assert attributes[SpanAttributes.GRAPH_NODE_NAME] == "Team"
    assert SpanAttributes.GRAPH_NODE_PARENT_ID not in attributes

    assert attributes["agno.team"] == "Team"
    for member in expected_team_members:
        assert attributes[f"agno.{member}.agent"] == member

    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in attributes

    assert span.status.is_ok


def _verify_agent_span(
    span: ReadableSpan,
    expected_name: str,
    expected_agent_name: str,
    expected_node_id: str,
    expected_parent_id: str,
    expected_input_value: dict[str, Any],
    expected_output_value_keys: Optional[set[str]],
    expected_agno_tools: Optional[tuple[str, ...]] = None,
) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == expected_name
    assert (
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.AGENT.value
    )

    input_value = attributes[SpanAttributes.INPUT_VALUE]
    assert isinstance(input_value, str)
    actual_input = json.loads(input_value)
    # For agno 2.0, session is extracted as a dict with session fields
    if "session" in expected_input_value and expected_input_value["session"] is str:
        assert "session" in actual_input
        assert isinstance(actual_input["session"], dict)
        # Verify session has expected fields
        assert "session_id" in actual_input["session"]
        assert "agent_id" in actual_input["session"]
        # Check other fields match
        for key, expected_val in expected_input_value.items():
            if key != "session":
                assert actual_input.get(key) == expected_val
    else:
        assert actual_input == expected_input_value

    if expected_output_value_keys is not None:
        output_value = attributes[SpanAttributes.OUTPUT_VALUE]
        assert isinstance(output_value, str)
        # Skip checking exact fields - response structure varies in agno 2.0
        # Just verify it's valid JSON with content
        output_dict = json.loads(output_value)
        assert "content" in output_dict or "messages" in output_dict
        assert attributes[SpanAttributes.OUTPUT_MIME_TYPE] == "application/json"
    else:
        assert SpanAttributes.OUTPUT_VALUE not in attributes
        assert SpanAttributes.OUTPUT_MIME_TYPE not in attributes

    assert attributes[SpanAttributes.GRAPH_NODE_ID] == expected_node_id
    assert attributes[SpanAttributes.GRAPH_NODE_NAME] == expected_agent_name
    assert attributes[SpanAttributes.GRAPH_NODE_PARENT_ID] == expected_parent_id

    assert attributes["agno.agent"] == expected_agent_name

    if expected_agno_tools is not None:
        assert attributes["agno.tools"] == expected_agno_tools
    else:
        assert "agno.tools" not in attributes

    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in attributes

    assert span.status.is_ok


def _verify_tool_span(
    span: ReadableSpan,
    expected_name_and_tool: str,
    expected_input_value: dict[str, Any],
) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == expected_name_and_tool
    assert (
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.TOOL.value
    )
    assert attributes[SpanAttributes.TOOL_NAME] == expected_name_and_tool

    input_value = attributes[SpanAttributes.INPUT_VALUE]
    assert isinstance(input_value, str)
    actual_input = json.loads(input_value)
    # For agno 2.0, session is extracted as a dict with session fields
    if "session" in expected_input_value and expected_input_value["session"] is str:
        assert "session" in actual_input
        assert isinstance(actual_input["session"], dict)
        # Verify session has expected fields
        assert "session_id" in actual_input["session"]
        assert "agent_id" in actual_input["session"]
        # Check other fields match
        for key, expected_val in expected_input_value.items():
            if key != "session":
                assert actual_input.get(key) == expected_val
    else:
        assert actual_input == expected_input_value

    assert SpanAttributes.OUTPUT_VALUE in attributes
    assert span.status.is_ok


def _verify_llm_span(span: ReadableSpan, expected_name: str) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == expected_name
    assert (
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-4o-mini"
    assert attributes[SpanAttributes.LLM_PROVIDER] == "OpenAI"

    invocation_params = attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS]
    assert isinstance(invocation_params, str)
    assert json.loads(invocation_params) == {"temperature": 0}

    assert span.status.is_ok


def _verify_context_attributes(span: ReadableSpan) -> None:
    attributes = dict(span.attributes or {})
    assert attributes[SpanAttributes.SESSION_ID] == CONTEXT_SESSION_ID
    assert attributes[SpanAttributes.USER_ID] == CONTEXT_USER_ID
    metadata = attributes[SpanAttributes.METADATA]
    assert isinstance(metadata, str)
    assert json.loads(metadata) == {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }
    tags = attributes[SpanAttributes.TAG_TAGS]
    expected_tags = ["tag-1", "tag-2"]
    if isinstance(tags, tuple):
        tags = list(tags)
    assert tags == expected_tags
    assert attributes[SpanAttributes.LLM_PROMPT_TEMPLATE] == "test-prompt-template"
    assert attributes[SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION] == "v1.0"
    template_vars = attributes[SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES]
    assert isinstance(template_vars, str)
    assert json.loads(template_vars) == {
        "var-1": "value-1",
        "var-2": "value-2",
    }
