"""Optional LLM client factory + structured LLM calls.

Tests must NOT call real LLMs. The recommended monkeypatch pattern is:

    from data_collection_workflow import llm_clients
    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", mock_fn)
    monkeypatch.setattr(llm_clients, "run_structured_llm_json", mock_fn)

Provider-specific dependencies (langchain_anthropic, langchain_openai) are
imported lazily inside `build_chat_model`, so importing this module does NOT
require those packages to be installed.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from .models import LLMExtractionOutput, LLMStructuredExtractionPolicy
from .llm_compatibility import anthropic_parameters, disallows_temperature, structured_output_method

OUTPUT_LANGUAGE_INSTRUCTION = (
    "Write generated explanations, summaries, and rationale in English. "
    "Preserve verbatim source quotations, source names, identifiers, and localized "
    "search queries in their original language. Do not translate evidence quotes "
    "or change factual values to satisfy the presentation language."
)


class LLMModelPreflightError(ValueError):
    """The selected model failed the initial API compatibility check."""


def _env_flag(name: str) -> bool:
    return (get_env(name) or "").strip().lower() == "true"


def llm_extraction_enabled() -> bool:
    """True only if ENABLE_LLM_EXTRACTION is set to "true" (case-insensitive)."""

    return _env_flag("ENABLE_LLM_EXTRACTION")


def llm_source_planning_enabled() -> bool:
    """True only if ENABLE_LLM_SOURCE_PLANNING is explicitly true."""

    return _env_flag("ENABLE_LLM_SOURCE_PLANNING")


def llm_source_critic_enabled() -> bool:
    """True only if ENABLE_LLM_SOURCE_CRITIC is explicitly true."""

    return _env_flag("ENABLE_LLM_SOURCE_CRITIC")


def llm_source_credibility_enabled() -> bool:
    """True only if ENABLE_LLM_SOURCE_CREDIBILITY is explicitly true."""

    return _env_flag("ENABLE_LLM_SOURCE_CREDIBILITY")


def llm_source_identity_enabled() -> bool:
    """True only if ENABLE_LLM_SOURCE_IDENTITY is explicitly true."""

    return _env_flag("ENABLE_LLM_SOURCE_IDENTITY")


def llm_disease_intelligence_enabled() -> bool:
    """True only if ENABLE_LLM_DISEASE_INTELLIGENCE is explicitly true."""

    return _env_flag("ENABLE_LLM_DISEASE_INTELLIGENCE")


def llm_fallback_to_rule_based() -> bool:
    """True unless LLM_FALLBACK_TO_RULE_BASED is explicitly "false"."""

    value = (get_env("LLM_FALLBACK_TO_RULE_BASED") or "").strip().lower()
    if value == "false":
        return False
    return True


def get_llm_settings() -> dict:
    """Read provider settings from the environment with safe defaults."""

    provider = (get_env("LLM_PROVIDER") or "anthropic").strip().lower()
    model = (get_env("LLM_MODEL") or "").strip()
    try:
        temperature = float(get_env("LLM_TEMPERATURE") or "0")
    except (TypeError, ValueError):
        temperature = 0.0
    try:
        max_tokens = int(get_env("LLM_MAX_TOKENS") or "4096")
    except (TypeError, ValueError):
        max_tokens = 4096
    try:
        timeout_seconds = float(get_env("LLM_TIMEOUT_SECONDS") or "0")
    except (TypeError, ValueError):
        timeout_seconds = 0.0
    try:
        max_retries = int(get_env("LLM_MAX_RETRIES") or "2")
    except (TypeError, ValueError):
        max_retries = 2
    return {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds if timeout_seconds > 0 else None,
        "max_retries": max(0, max_retries),
        "thinking": (get_env("LLM_THINKING") or "auto").strip().lower(),
        "effort": (get_env("LLM_EFFORT") or "").strip().lower() or None,
        "structured_output_method": (get_env("LLM_STRUCTURED_OUTPUT_METHOD") or "auto").strip().lower(),
    }


def _validate_provider_settings(settings: dict) -> None:
    provider = settings.get("provider")
    model = settings.get("model")
    if provider == "anthropic":
        if not model:
            raise ValueError(
                "LLM extraction enabled with provider=anthropic but LLM_MODEL "
                "is empty. Set LLM_MODEL to a Claude model name available in "
                "your Anthropic account."
            )
        if not get_env("ANTHROPIC_API_KEY"):
            raise ValueError(
                "LLM extraction enabled with provider=anthropic but ANTHROPIC_API_KEY "
                "is not set. Set ANTHROPIC_API_KEY to a valid Anthropic API key."
            )
        return
    if provider == "openai":
        if not model:
            raise ValueError(
                "LLM extraction enabled with provider=openai but LLM_MODEL is "
                "empty. Set LLM_MODEL to an OpenAI model name."
            )
        if not get_env("OPENAI_API_KEY"):
            raise ValueError(
                "LLM extraction enabled with provider=openai but OPENAI_API_KEY is "
                "not set. Set OPENAI_API_KEY to a valid OpenAI API key."
            )
        return
    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "Supported providers: 'anthropic', 'openai'."
    )


def chat_model_kwargs(settings: dict) -> dict:
    """Keep provider parameters and hard output caps consistent across stages."""
    provider = str(settings.get('provider') or '').lower()
    max_tokens = int(settings.get('max_tokens', 4096))
    if max_tokens <= 0:
        raise ValueError('max_tokens must be positive')
    kwargs = {'model': settings['model'], 'max_tokens': max_tokens,
              'max_retries': settings.get('max_retries', 2)}
    if settings.get('timeout_seconds'):
        kwargs['timeout'] = settings['timeout_seconds']
    if provider == 'anthropic':
        kwargs.update(anthropic_parameters(settings))
    elif provider == 'openai':
        if settings.get('thinking', 'auto') not in {None, 'auto'} or settings.get('effort'):
            raise ValueError('Claude thinking/effort controls require provider=anthropic; a compatible gateway needs its own API contract')
        if not disallows_temperature(settings.get('model')):
            kwargs['temperature'] = settings.get('temperature', 0.0)
    else:
        raise ValueError(f'Unsupported provider: {provider}')
    return kwargs


def build_chat_model(settings: dict | None = None):
    """Build a chat model for the configured provider. Does NOT invoke it."""

    settings = dict(settings or get_llm_settings())
    if get_env('PIPELINE_MODE') == 'evidence':
        settings['max_retries'] = 0  # Each external attempt must be budgeted explicitly.
    _validate_provider_settings(settings)
    provider = settings["provider"]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # local import keeps tests light

        return ChatAnthropic(**chat_model_kwargs(settings))
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # local import keeps tests light

        return ChatOpenAI(**chat_model_kwargs(settings))
    raise ValueError(f"Unsupported provider: {provider}")


def preflight_llm_model(settings: dict | None = None) -> dict:
    """Exercise the real extraction schema before long-running paid work.

    In evidence the runner calls this after acquisition checks with its session active;
    the single request is charged to model:preflight and is cached on resume.
    The synthetic observation is never merged into collection results.
    """
    from .session_runtime import BudgetExceeded, ResumeMismatch, ProviderAccountLimit
    resolved = {**get_llm_settings(), **(settings or {})}
    resolved['provider'] = str(resolved.get('provider') or '').strip().lower()
    resolved['model'] = str(resolved.get('model') or '').strip()
    try:
        chat_model = build_chat_model(resolved)
        messages = [
            {'role': 'system', 'content': 'This is a synthetic API compatibility check. Extract only explicitly supplied facts.'},
            {'role': 'user', 'content': 'Synthetic source: "There were 3 confirmed measles cases." Return exactly one record with disease="measles", cases_confirmed=3, evidence_quote="There were 3 confirmed measles cases.", and chunk_is_relevant=true. Leave other unsupported fields null or omitted.'},
        ]
        payload, mode = _invoke_structured_schema(
            chat_model, messages, LLMExtractionOutput, resolved,
            _langsmith_runnable_config('llm.preflight', stage='preflight', settings=resolved),
        )
        parsed = LLMExtractionOutput.model_validate(payload)
        if (not parsed.chunk_is_relevant or len(parsed.records) != 1
                or parsed.records[0].disease != 'measles'
                or parsed.records[0].cases_confirmed != 3
                or parsed.records[0].evidence_quote != 'There were 3 confirmed measles cases.'):
            raise ValueError('structured extraction preflight did not preserve the synthetic source facts')
    except (BudgetExceeded, ResumeMismatch, ProviderAccountLimit):
        raise
    except Exception as exc:
        raise LLMModelPreflightError(
            f"LLM model preflight failed for provider={resolved['provider']!r}, model={resolved['model']!r}: {type(exc).__name__}: {exc}"
        ) from exc
    return {'status': 'ok', 'provider': resolved['provider'], 'model': resolved['model'],
            'structured_output_method': mode, 'schema': 'LLMExtractionOutput'}


def _langsmith_runnable_config(run_name: str, *, stage: str, settings: dict | None = None) -> dict:
    settings = dict(settings or get_llm_settings())
    metadata = {
        "session_id": get_env("TRACE_SESSION_ID"),
        "trace_id": get_env("TRACE_ID"),
        "langsmith_project": get_env("LANGSMITH_PROJECT"),
        "llm_provider": settings.get("provider"),
        "llm_model": settings.get("model"),
        "workflow_stage": stage,
        "llm_settings": settings,
    }
    return {
        "run_name": run_name,
        "tags": ["data-collection-workflow", "data-collection-llm", stage],
        "metadata": {key: value for key, value in metadata.items() if value},
    }


def _token_usage_collector():
    from langchain_core.callbacks import BaseCallbackHandler
    from .session_runtime import provider_token_usage
    import threading
    class Collector(BaseCallbackHandler):
        def __init__(self):
            self.rows=[]
            self.unknown=False
            self.lock=threading.Lock()
        def on_llm_end(self,response,**kwargs):
            # One call-level aggregate; avoid counting aggregate plus per-message usage.
            usage=provider_token_usage(getattr(response,'llm_output',None) or {})
            if usage is None:
                rows=[provider_token_usage(getattr(g,'message',None)) for group in response.generations for g in group]
                if any(row is None or any(row.get(key) is None for key in ('input_tokens','output_tokens','total_tokens')) for row in rows):
                    with self.lock: self.unknown=True
                rows=[r for r in rows if r]
                if rows: usage={k:sum(r[k] for r in rows if r.get(k) is not None) if any(r.get(k) is not None for r in rows) else None for k in ('input_tokens','output_tokens','total_tokens')}
            with self.lock:
                if usage: self.rows.append(usage)
                else: self.unknown=True
        def on_llm_error(self,error,**kwargs):
            with self.lock: self.unknown=True
        def usage(self):
            with self.lock:
                if not self.rows: return None
                result={k:sum(r[k] for r in self.rows if r.get(k) is not None) if any(r.get(k) is not None for r in self.rows) else None for k in ('input_tokens','output_tokens','total_tokens')}
                if self.unknown or any(any(r.get(k) is None for k in result) for r in self.rows): result['availability']='partial'
                return result
    return Collector()


def _invoke_with_config(runnable: Any, messages: list, config: dict, *, validate_result=None) -> Any:
    # Normalize old caller metadata once; presence of a neutral key wins even
    # for empty/false values.  Do not mutate caller-owned configuration.
    metadata = dict(config.get('metadata') or {})
    for legacy, canonical in (
        ('hdc_stage', 'workflow_stage'), ('hdc_llm_settings', 'llm_settings'),
        ('hdc_chunk_id', 'chunk_id'), ('hdc_chunk_ids', 'chunk_ids'),
        ('hdc_recovery', 'recovery'),
    ):
        if legacy in metadata:
            metadata.setdefault(canonical, metadata.pop(legacy))
    config = {**config, 'metadata': metadata}
    def dispatch(invocation_config):
        result = _checked_invoke(runnable, messages, invocation_config,
                                 allow_schema_validation=validate_result is not None)
        if validate_result is not None:
            try:
                validated = validate_result(result)
            except (ValueError, TypeError) as exc:
                raise _response_contract_error(_short_validation_error(exc), result) from exc
            if isinstance(result, dict) and 'raw' in result and 'parsed' in result:
                result = {'raw': result['raw'], 'parsed': validated}
        return result

    if get_env('PIPELINE_MODE') == 'evidence':
        from .session_runtime import external_call
        stage = (config.get('metadata') or {}).get('workflow_stage') or 'unspecified'
        kind = 'extraction' if stage in {'structured_extraction', 'LLMExtractionOutput'} else 'model:' + stage
        model_obj = runnable
        for _ in range(4):
            child = getattr(model_obj, 'bound', None) or getattr(model_obj, 'first', None)
            if child is None:
                break
            model_obj = child
        identity = {key: getattr(model_obj, key, None) for key in ('model_name', 'model', 'temperature', 'max_tokens')}
        metadata=config.get('metadata') or {}
        collector=_token_usage_collector()
        invocation_config=dict(config)
        callbacks=config.get('callbacks')
        if callbacks is None or isinstance(callbacks,(list,tuple)):
            invocation_config['callbacks']=[*(callbacks or []),collector]
        else:
            manager=callbacks.copy()
            manager.add_handler(collector,inherit=True)
            invocation_config['callbacks']=manager
        return external_call(kind, {'messages': messages, 'stage': stage, 'model': identity, 'settings': metadata['llm_settings'] if 'llm_settings' in metadata else get_llm_settings(), 'chunk_id':metadata.get('chunk_id'), 'chunk_ids':metadata.get('chunk_ids') or []}, lambda: dispatch(invocation_config), recovery=bool(metadata.get('recovery')),usage_supplier=collector.usage,operation_metadata={k:metadata[k] for k in ('recovery_round','recovery_action_id') if k in metadata})
    return dispatch(config)


class LLMResponseContractError(ValueError):
    """A single received response failed validation; retain it for local recovery."""

    def __init__(self, message, raw_response):
        super().__init__(message)
        self.llm_raw_response = raw_response


def _response_contract_error(message, result):
    raw = result.get('raw') if isinstance(result, dict) and 'raw' in result and 'parsed' in result else result
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode='json')
    elif hasattr(raw, 'content'):
        raw = {'content': raw.content, 'response_metadata': getattr(raw, 'response_metadata', {})}
    # Keep response content out of ordinary error strings and console summaries.
    return LLMResponseContractError(str(message)[:400], json.loads(json.dumps(raw, default=str)))


def _short_validation_error(error):
    if isinstance(error, ValidationError):
        issues = error.errors(include_input=False, include_url=False)
        detail = '; '.join('.'.join(map(str, item['loc'])) + ':' + item['type'] for item in issues[:5])
        return 'Structured response failed schema validation: ' + detail
    return str(error)[:350] or type(error).__name__


def _checked_invoke(runnable: Any, messages: list, config: dict, *, allow_schema_validation=False) -> Any:
    """Inspect stop metadata before one local schema validation and ledger write."""
    result = _invoke_without_ledger(runnable, messages, config)
    raw = result.get('raw') if isinstance(result, dict) and 'raw' in result and 'parsed' in result else result
    metadata = getattr(raw, 'response_metadata', None) or {}
    reason = metadata.get('stop_reason') or metadata.get('finish_reason')
    if reason in {'max_tokens', 'length', 'refusal', 'content_filter', 'model_context_window_exceeded', 'pause_turn'}:
        raise _response_contract_error(f'LLM response stopped with {reason}; extraction is incomplete, not an empty result', result)
    if raw is not result:
        if not allow_schema_validation:
            if result.get('parsing_error') is not None:
                raise _response_contract_error('Structured response parsing failed', result)
            if result.get('parsed') is None:
                raise _response_contract_error('LLM response contained no parsed structured output', result)
        # SDK parsing can fail on a complete envelope serialized into one field.
        # The schema-aware validator inspects raw arguments before applying defaults.
        parsed = result.get('parsed')
        return {'raw': raw, 'parsed': parsed.model_dump() if isinstance(parsed, BaseModel) else parsed}
    from langchain_core.messages import BaseMessage
    if isinstance(result, BaseModel) and not isinstance(result, BaseMessage):
        return result.model_dump(exclude_unset=True)
    return result


def _invoke_without_ledger(runnable: Any, messages: list, config: dict) -> Any:
    # Select compatibility for lightweight callables before the only dispatch.
    # A provider TypeError after dispatch must never trigger an uncharged retry.
    import inspect
    try:
        signature = inspect.signature(runnable.invoke)
    except (TypeError, ValueError):
        return runnable.invoke(messages, config=config)
    try:
        signature.bind(messages, config=config)
    except TypeError:
        signature.bind(messages)
        return runnable.invoke(messages)
    return runnable.invoke(messages, config=config)


def _message_content(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(result)
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {None, "text", "output_text"}:
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(result)


def _strip_markdown_fence(text: str) -> str:
    stripped = (text or "").strip()
    fence_match = re.fullmatch(
        r"```(?:json|JSON)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _json_object_substrings(text: str) -> list[str]:
    substrings: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                substrings.append(text[start:index + 1])
                start = None
    return substrings


def _parse_json_object(text: str) -> dict:
    stripped = _strip_markdown_fence(text)
    if not stripped:
        raise ValueError("LLM returned empty output.")
    try:
        return _strict_json_object(stripped)
    except json.JSONDecodeError as original_error:
        # Prose around one complete JSON object is tolerated. Multiple answers
        # or duplicate keys are ambiguous and must never choose facts by size.
        candidates = []
        for candidate in _json_object_substrings(stripped):
            try:
                candidates.append(_strict_json_object(candidate))
            except json.JSONDecodeError:
                continue
        if len(candidates) > 1:
            raise ValueError('LLM returned multiple JSON objects; structured answer is ambiguous') from original_error
        if not candidates:
            raise original_error
        return candidates[0]


def _required_response_fields(schema_model):
    fields = schema_model.model_fields
    # These fields control whether work continues or evidence is treated as empty;
    # a provider must supply them, even when a legacy schema has defaults.
    controls = {'records', 'decision', 'query_batch'} & set(fields)
    return {name for name, field in fields.items() if field.is_required()} | controls


def _strict_json_object(text):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON object keys are ambiguous')
            result[key] = value
        return result
    value = json.loads(_strip_markdown_fence(text), object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError('Expected a complete JSON object envelope')
    return value


def _repair_serialized_envelope(payload, schema_model):
    """Unwrap at most one complete schema envelope; never reconstruct JSON tails."""
    fields = set(schema_model.model_fields)
    required = _required_response_fields(schema_model)
    needs_repair = not required <= set(payload)
    try:
        schema_model.model_validate(payload)
    except (ValueError, TypeError):
        needs_repair = True
    candidates = []
    for carrier, value in payload.items():
        if not needs_repair or carrier not in fields or not isinstance(value, str):
            continue
        try:
            inner = _strict_json_object(value)
        except (ValueError, TypeError):
            continue
        # A schema's text field may legitimately quote JSON. Only consider a
        # misplaced envelope when its outer field has the wrong shape or when
        # required top-level fields are absent, and all required data is inside.
        if carrier in inner and required <= set(inner) and set(inner) <= fields:
            candidates.append((carrier, inner))
    if len(candidates) > 1:
        raise ValueError('Multiple serialized schema envelopes are ambiguous')
    if candidates:
        carrier, inner = candidates[0]
        outer = {key: value for key, value in payload.items() if key != carrier}
        if any(key in inner and inner[key] != value for key, value in outer.items()):
            raise ValueError('Serialized schema envelope conflicts with outer fields')
        payload = {**outer, **inner}
    # A dangling JSON tail in prose is not a complete envelope. Reject before
    # defaults can turn missing decisions or query batches into successful stops.
    for name, value in payload.items():
        if not isinstance(value, str):
            continue
        try:
            _strict_json_object(value)
        except (ValueError, TypeError):
            pass
        else:
            continue  # An already valid text field may quote a complete JSON object.
        probe = value.replace('\\n', '\n')
        for sibling in fields - {name}:
            pattern = r'"\s*,\s*"' + re.escape(sibling) + r'"\s*:'
            if re.search(pattern, probe):
                raise ValueError(f'Embedded schema fields inside text field {name!r}')
    missing = required - set(payload)
    if missing:
        raise ValueError('Structured response missing explicit fields: ' + ', '.join(sorted(missing)))
    return payload


def _model_to_dict(result, schema_model: type[BaseModel]) -> dict:
    from langchain_core.messages import BaseMessage
    if isinstance(result, dict) and 'raw' in result and 'parsed' in result:
        raw = result['raw']
        calls = getattr(raw, 'tool_calls', None) or []
        if calls:
            matching = [call for call in calls if call.get('name') == schema_model.__name__]
            if len(matching) != 1 or len(calls) != 1:
                raise ValueError('Expected exactly one matching structured tool response')
            result = matching[0].get('args')
        elif hasattr(raw, 'content'):
            result = _parse_json_object(_message_content(raw))
        else:
            result = result['parsed']
    if isinstance(result, BaseMessage) or (hasattr(result, 'content') and not isinstance(result, BaseModel)):
        payload = _parse_json_object(_message_content(result))
    elif isinstance(result, BaseModel):
        payload = result.model_dump(exclude_unset=True)
    elif isinstance(result, dict):
        payload = result
    else:
        payload = _parse_json_object(_message_content(result))
    if not isinstance(payload, dict):
        raise ValueError('Structured response arguments must be a JSON object')
    payload = _repair_serialized_envelope(payload, schema_model)
    return schema_model.model_validate(payload).model_dump()


def run_structured_llm_json(
    system_prompt: str,
    user_prompt: str,
    expected_schema_name: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """Call the configured chat model and parse a JSON object response.

    This generic helper is intentionally small. Agent nodes use it only behind
    explicit feature flags, and tests monkeypatch it so no external API call is
    made during normal verification.
    """

    settings = get_llm_settings()
    if model:
        settings["model"] = model
    settings["temperature"] = temperature
    chat_model = build_chat_model(settings)
    messages = [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\n" + OUTPUT_LANGUAGE_INSTRUCTION
                + "\n\nReturn only one valid JSON object for schema: "
                + expected_schema_name
                + "."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    result = _invoke_with_config(
        chat_model,
        messages,
        _langsmith_runnable_config(
            f"llm.{expected_schema_name}",
            stage=expected_schema_name,
            settings=settings,
        ),
        validate_result=lambda value: value if isinstance(value, dict) else _parse_json_object(_message_content(value)),
    )
    if isinstance(result, dict):
        return result
    return _parse_json_object(_message_content(result))


def _invoke_structured_schema(chat_model, messages, schema_model, settings, config):
    """Select the wire format before the single provider dispatch; validate locally."""
    import inspect
    mode = structured_output_method(settings, schema_model)
    if mode == 'json_prompt':
        messages = [dict(message) for message in messages]
        instruction = ('\n\nReturn exactly one JSON object matching this schema. '
                       'Do not include reasoning or markdown in the answer. Omit unsupported optional fields.\n'
                       + json.dumps(schema_model.model_json_schema(), ensure_ascii=False))
        messages[0]['content'] += instruction
        runnable = chat_model
    else:
        method = chat_model.with_structured_output
        options = {'method': mode, 'include_raw': True}
        # Choose compatibility before dispatch for older local adapters. Never
        # retry a provider request because it raised TypeError after dispatch.
        try:
            inspect.signature(method).bind(schema_model, **options)
        except TypeError:
            if mode != 'function_calling':
                raise ValueError('Installed model adapter lacks native JSON schema support; update the adapter or use json_prompt')
            runnable = method(schema_model)
        else:
            runnable = method(schema_model, **options)
    invocation_config = {**config, 'metadata': {**(config.get('metadata') or {}),
                         'structured_output_method': mode, 'llm_settings': dict(settings)}}
    result = _invoke_with_config(
        runnable, messages, invocation_config,
        validate_result=lambda value: _model_to_dict(value, schema_model),
    )
    return _model_to_dict(result, schema_model), mode


def run_pydantic_structured_llm(
    system_prompt: str,
    user_prompt: str,
    schema_model: type[BaseModel],
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """One provider attempt with validated output, without hidden paid fallback."""
    settings = get_llm_settings()
    if model:
        settings['model'] = model
    settings['temperature'] = temperature
    chat_model = build_chat_model(settings)
    payload, mode = _invoke_structured_schema(
        chat_model,
        [{'role': 'system', 'content': system_prompt + '\n\n' + OUTPUT_LANGUAGE_INSTRUCTION},
         {'role': 'user', 'content': user_prompt}],
        schema_model, settings,
        _langsmith_runnable_config(f'llm.{schema_model.__name__}', stage=schema_model.__name__, settings=settings),
    )
    payload['_structured_output_mode'] = 'json_prompt' if mode == 'json_prompt' else 'provider_native'
    return payload


def _build_llm_messages(chunk: dict, policy: LLMStructuredExtractionPolicy) -> list:
    from .evidence_qualification import evidence_qualification_enabled
    universal = evidence_qualification_enabled()
    system_prompt = policy.system_prompt
    rules = list(policy.required_output_rules)
    if universal:
        system_prompt = system_prompt.replace(
            "Use the active target task disease and disease-intelligence terms supplied by the workflow context.",
            "Use the task only to select relevant evidence; preserve the disease explicitly stated by the source.",
        )
        # Replace inherited chunk-wide count instructions, rather than leaving
        # contradictory legacy and local-binding rules in the same prompt.
        replaced_prefixes = (
            "Set disease to the active task disease",
            "If a numeric death count is explicitly stated anywhere in the chunk",
            "If a numeric hospitalization count is explicitly stated",
        )
        rules = [rule for rule in rules if not rule.startswith(replaced_prefixes)]
        rules.extend([
            "Return evidence_quote as a short verbatim span from the current row or patient passage, never a paraphrase.",
            "Return field_provenance_json as an object mapping each populated fact field to {quote: exact supporting source text}. Use short field quotes, not the entire document.",
            "Source-bound context in bound_context_spans can supply an explicitly bound section heading, table header, or footnote. It belongs only to that row/section. Never borrow another row or patient's attributes.",
            "For metric_row_batch inputs, select source_row_id and use only that row's evidence text and its own bound_context_spans. Bracketed row IDs are transport labels, not source quotations.",
            "Task acceptance constraints, URL, publisher, search title, and workflow metadata do not prove disease, place, dates or counts. Leave unsupported fields null, including disease; do not replace a source disease with the task disease.",
            "Distinct metrics, case definitions and observation periods require separate records. Return empty records only when the supplied evidence contains no relevant observation; missing optional clinical fields do not discard an otherwise supported observation.",
            "Populate country and subnational_location independently from source evidence. Keep an explicitly supported locality, island, province or region even when its parent country is not stated. Do not infer the parent country from common geographic knowledge, the task, domain or institution; leave country null when unsupported. Preserve the source's geographic level rather than forcing it into a national total.",
            "Bind deaths, hospitalizations and other counts to the same observation as their disease, geography, population and observation period. A count elsewhere in the chunk, another country or another patient cannot populate this record. Extract another supported observation separately.",
            "Use metric_period_start and metric_period_end for an explicitly stated observation interval; preserve reporting_period and the source's date precision. Use event_start_date/event_end_date only for explicitly dated clinical event or outbreak beginning/ending, never for the interval over which cases were counted. Keep date_onset, date_confirmation and date_death attached to their own patient; publication_date, date_reported/report_date and as_of_date have distinct publication, reporting and cutoff roles and cannot substitute for one another.",
            "An observation interval does not by itself establish cumulative, annual or newly_reported count semantics. Populate statistical_count_type/count_semantics only when the source supports that meaning; otherwise use unknown or null. The requested task time window cannot enlarge the reported interval or establish an annual total.",
            "Keep evidence_quote as a complete local assertion that includes the count and its disease, geographic scope and observation interval when stated. For field_provenance_json, quote the supporting predicate or table cell with its explicit scope and time role, not an isolated number or date whose role is absent. Preserve exact source wording and do not append task or metadata text to the quotation.",
        ])
    system_text = system_prompt + "\n\n" + OUTPUT_LANGUAGE_INSTRUCTION + "\n\nRequired output rules:\n" + "\n".join(
        f"{i + 1}. {rule}" for i, rule in enumerate(rules)
    )
    user_parts = [
        f"source_id: {chunk.get('source_id')}",
        f"source_url: {chunk.get('source_url')}",
        f"source_type: {chunk.get('source_type')}",
        f"title: {chunk.get('title')}",
        f"publisher: {chunk.get('publisher')}",
        f"supporting_chunk_id: {chunk.get('chunk_id')}",
        f"fetch_purpose: {chunk.get('fetch_purpose')}",
        f"chunk_kind: {chunk.get('chunk_kind')}",
        f"data_types: {chunk.get('data_types')}",
        f"context_types: {chunk.get('context_types')}",
    ]
    if universal:
        if chunk.get("chunk_kind") == "metric_row_batch":
            evidence_units = chunk.get("metric_rows") or []
            user_parts.extend(["", "Source-bound rows (each row owns its bound_context_spans):",
                               json.dumps(evidence_units, ensure_ascii=False, sort_keys=True)])
        elif chunk.get("bound_context_spans"):
            user_parts.extend(["", "Source-bound bound_context_spans:",
                               json.dumps(chunk["bound_context_spans"], ensure_ascii=False, sort_keys=True)])
        if chunk.get("json_pointer") is not None:
            user_parts.extend(["", "Source JSON object pointer: " + str(chunk["json_pointer"])])
    if chunk.get("task_acceptance_contract"):
        user_parts.extend(
            [
                "",
                "Task acceptance contract:",
                json.dumps(
                    chunk.get("task_acceptance_contract"),
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            ]
        )
    user_parts.extend(
        [
            "",
            "Evidence chunk text:",
            chunk.get("text") or "",
        ]
    )
    user_text = "\n".join(user_parts)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def extract_chunk_with_llm(
    chunk: dict,
    policy: LLMStructuredExtractionPolicy,
) -> LLMExtractionOutput:
    """Call the configured LLM and return a validated LLMExtractionOutput.

    Raises on any provider/configuration/parse error so that the caller can
    decide whether to fall back to deterministic extraction.
    """

    settings = get_llm_settings()
    model = build_chat_model()
    messages = _build_llm_messages(chunk, policy)
    invocation_config = _langsmith_runnable_config("llm.LLMExtractionOutput", stage="structured_extraction")
    source_span_ids={str(row.get('chunk_id')) for row in chunk.get('metric_rows') or [] if row.get('chunk_id')}
    source_span_ids.update(str(key) for key in (chunk.get('row_chunk_id_by_id') or {}).values() if key)
    invocation_config['metadata'].update(chunk_id=chunk.get('parent_chunk_id') or chunk.get('chunk_id'),
                                         chunk_ids=sorted(source_span_ids),
                                         recovery=chunk.get('focused_recovery_status')=='focused_retry_attempted')
    payload, _ = _invoke_structured_schema(
        model, messages, LLMExtractionOutput, settings, invocation_config,
    )
    return LLMExtractionOutput.model_validate(payload)
