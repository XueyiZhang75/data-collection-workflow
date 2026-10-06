"""Controlled source-search providers for the data collection workflow."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .models import SearchProviderResponse, SearchResult

def _traceable_tool(name: str):
    try:
        from langsmith import traceable

        return traceable(name=name, run_type="tool")
    except Exception:
        return lambda fn: fn


class SearchProvider(Protocol):
    """Small provider interface: return search metadata, never page bodies."""

    provider: str

    def search(
        self,
        planned_query: dict,
        *,
        max_results: int,
        timeout_seconds: float,
    ) -> SearchProviderResponse:
        """Return normalized search metadata for one planned query."""


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Search fixture root must be an object: {path}")
    return data


def _result_from_dict(
    raw: dict,
    *,
    provider: str,
    planned_query: dict,
    rank: int,
) -> SearchResult:
    return SearchResult(
        title=raw.get("title"),
        url=raw.get("url"),
        snippet=raw.get("snippet") or raw.get("content") or raw.get("description"),
        published_date=raw.get("published_date") or raw.get("date"),
        source=raw.get("source") or raw.get("publisher") or raw.get("provider"),
        rank=int(raw.get("rank") or rank),
        query=planned_query.get("query"),
        query_id=planned_query.get("query_id"),
        provider_channel=planned_query.get("provider_channel"),
        source_type=planned_query.get("source_type"),
        role_hint=planned_query.get("role_hint"),
        retrieved_at=raw.get("retrieved_at"),
        provider=provider,
        query_type=planned_query.get("query_type"),
        raw=dict(raw),
    )


def _matches_fixture_block(block: dict, planned_query: dict) -> bool:
    query_id = str(planned_query.get("query_id") or "")
    query_text = str(planned_query.get("query") or "").lower()
    source_type = str(planned_query.get("source_type") or "")
    provider_channel = str(planned_query.get("provider_channel") or "")

    query_ids = {str(value) for value in block.get("query_ids") or []}
    if query_id and query_id in query_ids:
        return True

    source_types = {str(value) for value in block.get("source_types") or []}
    if source_type and source_type in source_types:
        return True

    provider_channels = {str(value) for value in block.get("provider_channels") or []}
    if provider_channel and provider_channel in provider_channels:
        return True

    match_terms = [
        str(value).lower()
        for value in block.get("match_terms") or []
        if str(value).strip()
    ]
    return bool(match_terms and any(term in query_text for term in match_terms))


class FixtureSearchProvider:
    """Deterministic local JSON search provider used by tests and smokes."""

    provider = "fixture"

    def __init__(self, fixture_path: str | Path):
        self.fixture_path = Path(fixture_path)

    def search(
        self,
        planned_query: dict,
        *,
        max_results: int,
        timeout_seconds: float,  # noqa: ARG002 - no network in fixture mode
    ) -> SearchProviderResponse:
        try:
            data = _load_json(self.fixture_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return SearchProviderResponse(
                provider=self.provider,
                query_id=planned_query.get("query_id"),
                query=planned_query.get("query"),
                results=[],
                raw_result_count=0,
                error=f"fixture_load_error:{exc}",
            )

        raw_results: list[dict] = []
        for block in data.get("queries") or []:
            if isinstance(block, dict) and _matches_fixture_block(block, planned_query):
                raw_results.extend(
                    item for item in block.get("results") or [] if isinstance(item, dict)
                )
        if not raw_results:
            raw_results = [
                item for item in data.get("results") or [] if isinstance(item, dict)
            ]

        limited = raw_results[: max(0, int(max_results))]
        results = [
            _result_from_dict(
                item,
                provider=self.provider,
                planned_query=planned_query,
                rank=index,
            )
            for index, item in enumerate(limited, start=1)
        ]
        return SearchProviderResponse(
            provider=self.provider,
            query_id=planned_query.get("query_id"),
            query=planned_query.get("query"),
            results=results,
            raw_result_count=len(raw_results),
            warnings=list(data.get("warnings") or []),
        )


class TavilySearchProvider:
    """Small Tavily adapter for bounded live search metadata."""

    provider = "tavily"

    def __init__(self, api_key_env: str = "TAVILY_API_KEY"):
        self.api_key_env = api_key_env

    @_traceable_tool("tavily_search")
    def search(
        self,
        planned_query: dict,
        *,
        max_results: int,
        timeout_seconds: float,
    ) -> SearchProviderResponse:
        query = str(planned_query.get("query") or "").strip()
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return SearchProviderResponse(
                provider=self.provider,
                query_id=planned_query.get("query_id"),
                query=query,
                results=[],
                raw_result_count=0,
                error=f"missing_api_key:{self.api_key_env}",
            )

        payload = json.dumps(
            {
                "query": query,
                "search_depth": "basic",
                "max_results": max(1, int(max_results)),
                "include_answer": False,
                "include_raw_content": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=float(timeout_seconds),
            ) as response:
                body = response.read().decode("utf-8")
            # Stamp only this actual response, before it can enter the session cache.
            retrieved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return SearchProviderResponse(
                provider=self.provider,
                query_id=planned_query.get("query_id"),
                query=query,
                results=[],
                raw_result_count=0,
                error=f"provider_error:{exc.__class__.__name__}",
            )

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            return SearchProviderResponse(
                provider=self.provider,
                query_id=planned_query.get("query_id"),
                query=query,
                results=[],
                raw_result_count=0,
                error=f"invalid_provider_json:{exc.__class__.__name__}",
            )

        raw_results = [
            item for item in data.get("results") or [] if isinstance(item, dict)
        ]
        results = [
            _result_from_dict(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("content"),
                    "published_date": item.get("published_date"),
                    "source": item.get("source") or "Tavily",
                    "rank": index,
                    "provider": self.provider,
                    "retrieved_at": retrieved_at,
                },
                provider=self.provider,
                planned_query=planned_query,
                rank=index,
            )
            for index, item in enumerate(raw_results[: max(0, int(max_results))], start=1)
        ]
        return SearchProviderResponse(
            provider=self.provider,
            query_id=planned_query.get("query_id"),
            query=query,
            results=results,
            raw_result_count=len(raw_results),
            warnings=[],
        )


def build_search_provider(
    *,
    provider: str,
    mode: str,
    fixture_path: str | Path | None = None,
) -> SearchProvider:
    """Return a configured search provider without performing any search."""

    provider_name = (provider or "").strip().lower()
    mode_name = (mode or "").strip().lower()
    if mode_name == "fixture" or provider_name == "fixture":
        if not fixture_path:
            raise ValueError("fixture search mode requires a fixture_path")
        return FixtureSearchProvider(fixture_path)
    if provider_name in {"", "tavily"}:
        return TavilySearchProvider()
    raise ValueError(f"Unsupported search provider: {provider}")


def search_api_key_present(provider: str) -> bool:
    provider_name = (provider or "").strip().lower()
    if provider_name == "tavily":
        return bool(os.environ.get("TAVILY_API_KEY"))
    if provider_name == "brave":
        return bool(os.environ.get("BRAVE_SEARCH_API_KEY"))
    if provider_name == "serpapi":
        return bool(os.environ.get("SERPAPI_API_KEY"))
    if provider_name == "bing":
        return bool(os.environ.get("BING_SEARCH_V7_SUBSCRIPTION_KEY"))
    return False
