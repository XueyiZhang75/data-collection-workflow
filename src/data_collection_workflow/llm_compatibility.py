"""Provider wire compatibility; model IDs remain user-configurable.

Rules checked against Claude Platform migration/structured-output documentation
on 2026-09-22. Unknown IDs retain the legacy tool path; preflight verifies access.
This module never changes the selected model or expands an output budget.
"""
from __future__ import annotations

import re


def _matches(model, pattern):
    return bool(re.match(pattern + r'(?:$|[-_])', str(model or '').lower()))


def disallows_temperature(model):
    return (_matches(model, r'claude-(?:fable|mythos|sonnet|opus)-5')
            or _matches(model, r'claude-opus-4-[78]')
            or _matches(model, r'claude-mythos-preview'))


def disallows_forced_tools(model):
    return (_matches(model, r'claude-(?:fable|mythos)-5-1')
            or _matches(model, r'claude-opus-5-5'))


def anthropic_parameters(settings):
    model = settings.get('model')
    thinking = settings.get('thinking') or 'auto'
    effort = settings.get('effort') or None
    if thinking not in {'auto', 'adaptive', 'disabled'}:
        raise ValueError('thinking must be auto, adaptive, or disabled')
    if effort not in {None, 'low', 'medium', 'high', 'xhigh', 'max'}:
        raise ValueError('unsupported Claude effort value')
    adaptive_supported = disallows_temperature(model) or _matches(model, r'claude-(?:opus|sonnet)-4-6')
    if thinking == 'adaptive' and not adaptive_supported:
        raise ValueError(f'{model} has no known adaptive thinking support; use thinking=auto')
    if thinking == 'disabled' and (disallows_forced_tools(model) or (
        _matches(model, r'claude-opus-5') and effort in {'xhigh', 'max'}
    )):
        raise ValueError(f'{model} does not allow thinking=disabled with this effort')
    opus45 = _matches(model, r'claude-opus-4-5')
    if effort and not (adaptive_supported or opus45):
        raise ValueError(f'{model} has no known effort support; omit effort')
    if opus45 and effort in {'max', 'xhigh'}:
        raise ValueError(f'{model} does not support {effort} effort')
    if effort == 'xhigh' and (_matches(model, r'claude-(?:opus|sonnet)-4-6')
                              or _matches(model, r'claude-mythos-preview')):
        raise ValueError(f'{model} does not support xhigh effort')
    result = {}
    if thinking != 'auto':
        result['thinking'] = {'type': thinking}
    if effort:
        result['output_config'] = {'effort': effort}
    if not disallows_temperature(model) and thinking != 'adaptive':
        result['temperature'] = settings.get('temperature', 0.0)
    return result


def native_schema_limits(schema_model):
    """Conservative precheck; unrestricted dictionaries cannot be closed safely.

    Pydantic's fixed-property objects can be closed by the provider SDK. Dynamic
    dictionary keys cannot. Definitions are counted once, not expanded per $ref.
    """
    optional = unions = 0
    open_objects = False
    def visit(value):
        nonlocal optional, unions, open_objects
        if isinstance(value, dict):
            properties = value.get('properties') or {}
            optional += len(set(properties) - set(value.get('required') or []))
            unions += int('anyOf' in value) + int('oneOf' in value)
            if value.get('type') == 'object':
                additional = value.get('additionalProperties')
                if additional is True or isinstance(additional, dict) or not properties:
                    open_objects = True
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(schema_model.model_json_schema())
    return {'optional': optional, 'unions': unions, 'open_objects': open_objects,
            'compatible': optional <= 24 and unions <= 16 and not open_objects}


def structured_output_method(settings, schema_model):
    requested = settings.get('structured_output_method') or 'auto'
    if requested not in {'auto', 'function_calling', 'json_schema', 'json_prompt'}:
        raise ValueError('structured_output_method must be auto, function_calling, json_schema, or json_prompt')
    no_force = disallows_forced_tools(settings.get('model'))
    native = settings.get('provider') == 'anthropic'
    limits = native_schema_limits(schema_model)
    if requested == 'auto':
        # Modern models can serialize large tool envelopes into a single field.
        # Pick the compatible format before dispatch instead of retrying a failed call.
        if native and disallows_temperature(settings.get('model')) and not limits['compatible']:
            return 'json_prompt'
        if no_force:
            return 'json_schema' if native and limits['compatible'] else 'json_prompt'
        # Be explicit: the OpenAI adapter's default is JSON schema, whereas the
        # project's large schemas were designed for non-strict tool calling.
        return 'function_calling'
    if requested == 'function_calling' and no_force:
        raise ValueError('This Claude model rejects forced tools; use structured_output_method=auto or json_prompt')
    if requested == 'json_schema' and native and not limits['compatible']:
        raise ValueError(f'Claude strict JSON schema exceeds supported limits ({limits}); use auto or json_prompt')
    return requested
