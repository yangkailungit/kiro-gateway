# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read AGENTS.md first

`AGENTS.md` is the authoritative contributor guide: project philosophy, code conventions (English-only identifiers, mandatory type hints, Google-style docstrings, no bare `except`), the testing philosophy, and commit message format (`<type>(<scope>): <description> (#issue)`). This file covers only the commands and the architecture that aren't obvious from a single file.

Caveat: `AGENTS.md` and `docs/*/ARCHITECTURE.md` predate some changes. `kiro/config.py` is the single source of truth — e.g. the Kiro API host is now `https://runtime.{region}.kiro.dev`, not the `codewhisperer.*` host the docs still list, and `APP_VERSION` lives in `config.py` (re-exported as `kiro.__version__`).

## Commands

```bash
pip install -r requirements.txt        # deps (prod + test in one file)
python main.py                         # run; --host/--port override env
pytest                                 # full suite (pythonpath=. via pytest.ini)
pytest tests/unit/test_auth_manager.py -v
pytest tests/unit/test_auth_manager.py::TestKiroAuthManagerInitialization::test_initialization_stores_credentials -v
pytest --cov=kiro --cov-report=html    # needs pytest-cov
python manual_api_test.py              # hits the real API; excluded from pytest
```

No linter or formatter is configured — don't introduce one unprompted. CI (`.github/workflows/docker.yml`) runs `pytest -v --tb=short`, then coverage, then the Docker build with a Trivy scan.

## Architecture: shared core, thin adapters

Two client-facing API surfaces sit on one shared core. Anything touching request/response handling has **four** code paths that must stay consistent — OpenAI and Anthropic, streaming and non-streaming. Partial changes are treated as broken changes (see AGENTS.md principle 10).

```
routes_openai.py     routes_anthropic.py       ← FastAPI, API-key gate, failover loop
converters_openai.py converters_anthropic.py   ← client format → UnifiedMessage/UnifiedTool
              converters_core.py               ← Kiro payload building (all real logic)
                        ↓
    auth.py · http_client.py · account_manager.py · cache.py · model_resolver.py
                        ↓  POST /generateAssistantResponse (AWS event stream)
              streaming_core.py                ← parse to KiroEvent objects
streaming_openai.py  streaming_anthropic.py    ← KiroEvent → SSE in each dialect
```

The adapters normalize into `UnifiedMessage`/`UnifiedTool` (`converters_core.py`) and then call `build_kiro_payload`. New conversion logic belongs in the core; only format-specific extraction belongs in an adapter. Non-streaming responses are produced by consuming the same stream (`collect_stream_to_result`), not by a separate path.

### Kiro API quirk workarounds live in converters_core.py

Kiro rejects many message shapes with a single vague "Improperly formed request". `converters_core.py` fixes these classes of problem in sequence: `sanitize_json_schema`, `validate_tool_names`, `normalize_message_roles`, `ensure_first_message_is_user`, `ensure_assistant_before_tool_results`, `merge_adjacent_messages`, `ensure_alternating_roles`. Long tool descriptions (> `TOOL_DESCRIPTION_MAX_LENGTH`) are replaced by a pointer and the full text is appended to the system prompt as `## Tool: {name}` (`process_tools_with_long_descriptions`). When you hit a new 400 from Kiro, add a named transformation here rather than an inline conditional.

### Account System (multi-account failover)

Off by default; `ACCOUNT_SYSTEM=true` enables it. `credentials.json` lists accounts (`type` of `json`/`sqlite`/`refresh_token`; a `path` pointing at a directory is scanned and validated file-by-file), and `state.json` persists the sticky index and per-account failure state.

Each `Account` owns its own `KiroAuthManager`, `ModelInfoCache`, and `ModelResolver`, all lazily initialized. `AccountManager` keeps one **global** sticky index (not per-model) and a Circuit Breaker keyed on consecutive failures with exponential backoff plus a probabilistic retry chance.

The failover loop lives in the route handlers. Errors are classified by `account_errors.classify_error` into `FATAL` (return to the client immediately) or `RECOVERABLE` (report the failure, try the next account, up to `len(accounts) * 2` attempts). Network errors surface as HTTP 502/504 from `request_with_retry` and count as RECOVERABLE. With exactly one account the loop breaks instead of retrying, and the original status code is preserved. When the system is disabled, routes fall back to legacy single-account mode via `get_first_account()`.

### Model resolution

`model_resolver.py` runs: normalize the client name (dashes to dots, strip date suffixes) → dynamic cache from `/ListAvailableModels` → `HIDDEN_MODELS`/`MODEL_ALIASES` in config → pass through unknown names to Kiro. The gateway is not a gatekeeper; Kiro is the final arbiter of validity.

### Token counts are derived, not reported

Kiro returns only `contextUsagePercentage`. `total_tokens` comes from that percentage times the model's max input tokens, `completion_tokens` from tiktoken (`cl100k_base` with a 1.15 Claude correction factor), and `prompt_tokens` is the difference.

### Other cross-cutting systems

- **Truncation recovery** (`truncation_state.py` + `truncation_recovery.py`): a mid-JSON stream cut is detected in `parsers.py`, cached, and turned into a synthetic follow-up message on the next request. Toggle: `TRUNCATION_RECOVERY`.
- **WebSearch** (`mcp_tools.py`): Path A is a native Anthropic `web_search*` tool type, which early-returns straight to the MCP API and bypasses failover; Path B injects a `web_search` tool and emulates SSE. Path A works regardless of `WEB_SEARCH_ENABLED`.
- **Fake reasoning / thinking** (`thinking_parser.py`, `FAKE_REASONING_*`): an FSM extracts `<thinking>`-style blocks out of the text stream and re-emits them as reasoning content.
- **Payload guards** (`payload_guards.py`): enforce `KIRO_MAX_PAYLOAD_BYTES`; trimming happens only with `AUTO_TRIM_PAYLOAD=true`, since deciding what to drop is the client's call.
- **Debug logging** (`debug_logger.py`, `debug_middleware.py`): `DEBUG_MODE` is `off`/`errors`/`all`, writing request and both raw and transformed streams into `debug_logs/`.

## Streaming gotcha

Streaming must use a per-request `httpx.AsyncClient` (`async with`); reusing the shared pooled client leaks CLOSE_WAIT sockets. Non-streaming requests use the shared client. `streaming_core.stream_with_first_token_retry` retries when the first token doesn't arrive within `FIRST_TOKEN_TIMEOUT`, and `initial_response` is threaded through so a successful retry's response isn't re-requested.

## Tests

Every change ships with tests (AGENTS.md principle 7). `tests/conftest.py` applies `block_all_network_calls` globally — any real httpx connection fails the test, so the suite must pass offline. Find the right existing `test_*.py` via the map in `tests/README.md` instead of adding files; classes are `Test*Success` / `Test*Errors` / `Test*EdgeCases`, tests follow Arrange-Act-Assert, and async tests need `@pytest.mark.asyncio`.
