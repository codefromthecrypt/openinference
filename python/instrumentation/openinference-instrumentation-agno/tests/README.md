# Agno Instrumentation Tests

## Debugging and Logging

### How to Enable Debug Logging for Instrumentation

When debugging instrumentation issues, you MUST be able to see debug output from the wrappers. This is CRITICAL for understanding what's happening during test execution.

#### Method 1: Using tox (REQUIRED FOR INSTRUMENTATION DEBUGGING)
```bash
# This WILL show debug output from instrumentation
uvx --with tox-uv tox -e py313-ci-agno -- -xvs tests/test_instrumentor.py::test_agno_instrumentation

# The key parts:
# - tox will rebuild and reinstall the package in editable mode
# - -xvs enables verbose output with stdout capture disabled
# - Debug logging is configured in conftest.py fixture
```

#### Method 2: Direct pytest (WILL NOT SHOW INSTRUMENTATION LOGS)
```bash
# This will NOT show instrumentation debug output if package isn't installed in editable mode
pytest tests/test_instrumentor.py::test_agno_instrumentation -xvs
```

### Important Configuration Files

1. **tests/conftest.py** - Sets up logging in the `instrument` fixture:
```python
@pytest.fixture(autouse=True)
def instrument() -> Generator[AgnoInstrumentor, None, None]:
    import logging
    import sys

    # Configure logging for instrumentation modules
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Explicitly set the loggers we care about
    logging.getLogger("openinference.instrumentation.agno").setLevel(logging.DEBUG)
    logging.getLogger("openinference.instrumentation.agno._wrappers").setLevel(logging.DEBUG)
```

2. **tests/pytest.ini** - Enables pytest CLI logging:
```ini
[pytest]
addopts = -v
log_cli = true
log_cli_level = DEBUG
```

3. **src/openinference/instrumentation/agno/__init__.py** - Module logger setup:
```python
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

### Common Pitfalls (NEVER DO THESE)

1. **Using grep/head with test output**: NEVER use `| grep` or `| head` when debugging - it filters out debug messages
2. **Package not installed in editable mode**: If you modify the instrumentation code but don't reinstall, your changes won't be reflected
3. **Using direct pytest instead of tox**: Direct pytest may use a cached/installed version instead of your local changes
4. **Writing debug output to files instead of using logging**: This is abnormal and unnecessary

### What You Should See

When logging is properly configured, you should see:
- Fixture setup messages: `========= FIXTURE: Creating AgnoInstrumentor =========`
- Instrumentation setup: `AGNO INSTRUMENTOR _instrument CALLED`
- Wrapper invocations: `WRAPPER RUN CALLED - Instance: Team, Name: Web Scraping Team, Method: _run`
- Detailed span creation logs

### Troubleshooting

If you don't see debug output:
1. Use `tox -r` to recreate the environment (forces reinstall)
2. Check that the package is installed from local source, not PyPI
3. Verify logging configuration in conftest.py
4. Ensure you're NOT filtering output with grep/head
5. Run with `-xvs` flags to disable output capture

### Example Debug Session

```bash
# 1. Make changes to instrumentation code
vim src/openinference/instrumentation/agno/_wrappers.py

# 2. Run tests with tox to ensure package is rebuilt (REQUIRED)
uvx --with tox-uv tox -e py313-ci-agno -- -xvs tests/test_instrumentor.py::test_agno_instrumentation

# 3. Save full output to file if needed (but DON'T use grep/head during test run)
uvx --with tox-uv tox -e py313-ci-agno -- -xvs tests/test_instrumentor.py::test_agno_instrumentation 2>&1 | tee /tmp/test_output.txt

# 4. Search the saved file AFTER the test completes
grep "WRAPPER" /tmp/test_output.txt
```

## Re-recording VCR Cassettes

When tests fail due to outdated VCR cassettes (e.g., API authentication errors or changed responses), follow these steps to re-record:

### Prerequisites
1. Ensure `OPENAI_API_KEY` is set in your environment with a valid API key
2. The `passenv = OPENAI_API_KEY` directive must be present in the root `tox.ini` file

### Steps to Re-record

1. Delete the existing cassette file:
```bash
rm tests/cassettes/test_agno_instrumentation.yaml
```

2. Run the tests with VCR in record mode using tox:
```bash
OPENAI_API_KEY=$OPENAI_API_KEY uvx --with tox-uv tox -r -e py313-ci-agno -- tests/test_instrumentor.py::test_agno_instrumentation -xvs --vcr-record=once
```

### Important Notes
- The test reads `OPENAI_API_KEY` from the environment, falling back to "sk-test" if not set
- VCR will cache responses including authentication errors (401), so always delete the cassette before re-recording
- The `--vcr-record=once` flag ensures the cassette is only recorded when it doesn't exist
- Use `-r` flag with tox to ensure a clean environment when re-recording
- The tests use `SinglePageWebsiteTools` which doesn't require API keys, and works around a redirect bug in `WebsiteTools`
