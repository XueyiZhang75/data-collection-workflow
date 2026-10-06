"""Chromium transport with reservations before every supported HTTP dispatch."""
from __future__ import annotations

import base64
import json
import re
from urllib.parse import urlsplit
from uuid import uuid4

from .acquisition_budget import is_adaptive_budget
from .session_runtime import BudgetExceeded, get_runtime


_FETCH_PATTERNS = [{"urlPattern": "*", "requestStage": "Request"},
                   {"urlPattern": "*", "requestStage": "Response"}]


class _RequestGate:
    """Use public CDP sessions; child iframe sessions use nested CDP messages."""

    def __init__(self, context, page, url, config):
        self.context = context
        self.page = page
        self.session = context.new_cdp_session(page)
        self.runtime = get_runtime()
        self.account = self.runtime is not None and is_adaptive_budget(getattr(self.runtime, "config", {}) or {})
        self.source_target = config.get("_source_target_id") or url
        self.navigation_id = uuid4().hex
        self.sequence = 0
        self.command_id = 0
        self.pending_commands = {}
        self.redirect_depths = {}
        self.targets = set()
        self.budget_exhausted_kind = None
        self.request_errors = []
        self.unsupported_contexts = []
        self.closing = False

    def _unsupported(self, kind, url=None):
        detail = {"kind": kind, **({"url": url} if url else {})}
        if detail not in self.unsupported_contexts:
            self.unsupported_contexts.append(detail)
            self.request_errors.append({"reason": "unsupported_browser_context", **detail})

    def _command_error(self, method, error):
        if not self.closing:
            self.request_errors.append({"reason": "browser_protocol_error", "method": method,
                                        "error": str(error)})

    def _send(self, path, method, params=None, completed=None):
        """Forward only via public Target.sendMessageToTarget, including OOPIFs."""
        params = params or {}
        if not path:
            result = self.session.send(method, params)
            if completed:
                completed(result)
            return
        self.command_id += 1
        ident = self.command_id
        self.pending_commands[(path, ident)] = (method, completed)
        message = {"id": ident, "method": method, "params": params}
        for session_id in reversed(path):
            self.command_id += 1
            message = {"id": self.command_id, "method": "Target.sendMessageToTarget",
                       "params": {"sessionId": session_id, "message": json.dumps(message)}}
        try:
            self.session.send(message["method"], message["params"])
        except Exception:
            self.pending_commands.pop((path, ident), None)
            raise

    def _attach(self, path, params):
        info = params["targetInfo"]
        target = info["targetId"]
        if target in self.targets:
            return
        self.targets.add(target)
        child = (*path, params["sessionId"])
        kind = info.get("type")
        if kind not in {"iframe", "worker"}:
            self._unsupported(kind or "unknown_target", info.get("url"))
            # Keep an unsupported newly attached execution context paused.
            return
        # Worker HTTP traffic is intercepted by its owning page; the response
        # CSP blocks handshakes outside that gate before its script executes.
        if kind == "worker":
            self._send(child, "Runtime.runIfWaitingForDebugger")
            return
        commands = [("Fetch.enable", {"patterns": _FETCH_PATTERNS}),
                    ("Network.setCacheDisabled", {"cacheDisabled": True}),
                    ("Log.enable", {})]
        commands += [("Target.setAutoAttach", {"autoAttach": True,
                      "waitForDebuggerOnStart": True, "flatten": False}),
                     ("Runtime.runIfWaitingForDebugger", {})]

        def configure(index=0):
            if self.closing or index >= len(commands):
                return
            method, values = commands[index]
            try:
                def configured(value):
                    if isinstance(value, dict) and value.get("exceptionDetails"):
                        self._command_error(method, value["exceptionDetails"])
                        self._unsupported(kind, info.get("url"))
                        return
                    configure(index + 1)
                self._send(child, method, values, completed=configured)
            except Exception as exc:
                self._command_error(method, exc)
                self._unsupported(kind, info.get("url"))
        configure()

    def _abort(self, path, request_id):
        try:
            self._send(path, "Fetch.failRequest", {"requestId": request_id, "errorReason": "Aborted"})
        except Exception as exc:
            self._command_error("Fetch.failRequest", exc)

    def _paused(self, path, params):
        request = params["request"]
        request_id = params["requestId"]
        if params.get("responseErrorReason"):
            self.request_errors.append({"reason": "request_failed", "url": request["url"],
                                        "error": params["responseErrorReason"]})
            self._abort(path, request_id)
            return
        if "responseStatusCode" in params:
            # Apply a connection policy to documents and worker responses.
            # Worker script requests can be classified as Other, and blob
            # workers inherit their document policy. Preserve bytes/SRI and
            # any stricter original CSP; no ws/wss handshake may bypass billing.
            headers = [*(params.get("responseHeaders") or []),
                       {"name": "Content-Security-Policy", "value": "connect-src http: https: data: blob:"}]
            try:
                self._send(path, "Fetch.continueResponse", {"requestId": request_id,
                    "responseCode": params["responseStatusCode"], "responseHeaders": headers})
            except Exception as exc:
                self._command_error("Fetch.continueResponse", exc)
                self._unsupported("unconfigured_script_connection_policy", request["url"])
                self._abort(path, request_id)
            return
        if self.closing or self.budget_exhausted_kind:
            self._abort(path, request_id)
            return
        url = request["url"]
        previous = params.get("redirectedRequestId")
        depth = 0
        if previous:
            prior_depth = self.redirect_depths.get((path, previous))
            if prior_depth is None:
                self.request_errors.append({"reason": "redirect_chain_untracked", "url": url})
                self._abort(path, request_id)
                return
            depth = prior_depth + 1
        self.redirect_depths[(path, request_id)] = depth
        if depth > 5:
            # Match the native transport: initial request plus five redirects.
            # Reject before reserving HTTP usage or dispatching the sixth hop.
            self.request_errors.append({"reason": "redirect_limit_exceeded", "url": url,
                                        "redirects": depth, "limit": 5})
            self._abort(path, request_id)
            return
        self.sequence += 1
        def dispatch():
            self._send(path, "Fetch.continueRequest", {"requestId": request_id})
            # This operation records dispatch, not a reusable HTTP response.
            return {"dispatched_url": url, "method": request.get("method", "GET")}
        try:
            if self.account and urlsplit(url).scheme.lower() in {"http", "https"}:
                self.runtime.call("http_request", {
                    "navigation_id": self.navigation_id, "sequence": self.sequence,
                    "request_id": request_id, "session_path": path, "url": url,
                    "method": request.get("method", "GET")}, dispatch,
                    source_target=self.source_target,
                    operation_metadata={"transport": "chromium", "resource_type": params.get("resourceType")})
            else:
                dispatch()
        except BudgetExceeded as exc:
            self.budget_exhausted_kind = exc.kind
            self._abort(path, request_id)
        except Exception as exc:
            self.request_errors.append({"reason": "browser_request_error", "url": url, "error": str(exc)})
            self._abort(path, request_id)

    def _event(self, path, message):
        if self.closing:
            return
        method = message.get("method")
        params = message.get("params") or {}
        if method == "Target.receivedMessageFromTarget":
            self._event((*path, params["sessionId"]), json.loads(params["message"]))
        elif method == "Target.attachedToTarget":
            self._attach(path, params)
        elif method == "Fetch.requestPaused":
            self._paused(path, params)
        elif method == "Log.entryAdded":
            text = (params.get("entry") or {}).get("text") or ""
            if "connect-src" in text and re.search(r"\bwss?://", text):
                self._unsupported("websocket")
        elif "id" in message:
            command = self.pending_commands.pop((path, message["id"]), None)
            if command:
                name, completed = command
                if message.get("error"):
                    self._command_error(name, message["error"])
                    # Do not resume a child whose interception setup failed.
                    if completed:
                        self._unsupported("unconfigured_child_target")
                elif completed:
                    completed(message.get("result"))

    def install(self):
        for name in ("Target.receivedMessageFromTarget", "Target.attachedToTarget", "Fetch.requestPaused", "Log.entryAdded"):
            self.session.on(name, lambda params, name=name: self._event((), {"method": name, "params": params}))
        self.session.send("Fetch.enable", {"patterns": _FETCH_PATTERNS})
        self.session.send("Network.setCacheDisabled", {"cacheDisabled": True})
        self.session.send("Log.enable", {})
        self.session.send("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": False})

        def guard(route):
            # Context routing protects new windows before their first request.
            # It is not used for HTTP billing: it misses redirect destinations.
            try:
                supported = route.request.frame.page == self.page
            except Exception:
                supported = False
            if not supported:
                self._unsupported("unbound_page_request", route.request.url)
                route.abort("blockedbyclient")
            else:
                route.continue_()
        self.context.route("**/*", guard)
        def block_socket(socket):
            # Fetch interception excludes WebSocket upgrade handshakes. A
            # routed socket has no server connection until connect_to_server.
            self._unsupported("websocket", socket.url)
            # Returning without connect_to_server keeps this socket local.
            # The browser context owns its lifetime and closes it on disposal.
        self.context.route_web_socket("**/*", block_socket)
        self.context.expose_binding("__collection_unsupported_context", lambda _source, kind: self._unsupported(kind))
        self.context.add_init_script("""(() => {
            const report = globalThis.__collection_unsupported_context;
            if (typeof globalThis.SharedWorker !== 'undefined') {
                Object.defineProperty(globalThis, 'SharedWorker', {
                    configurable: false, writable: false,
                    value: function () {
                        report('shared_worker');
                        throw new Error('unsupported_browser_context: shared_worker');
                    }
                });
            }
        })();""")


def browser_fetch(url, config):
    """Return rendered HTML and data responses; caller owns navigation caching."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as playwright:
        options = {"headless": True}
        if config.get("chromium_executable"):
            options["executable_path"] = config["chromium_executable"]
        browser = playwright.chromium.launch(**options)
        gate = None
        try:
            context = browser.new_context(service_workers="block")
            page = context.new_page()
            gate = _RequestGate(context, page, url, config)
            gate.install()
            context.on("requestfailed", lambda request: gate.request_errors.append({
                "reason": "request_failed", "url": request.url, "error": request.failure}))
            def console_error(message):
                if "connect-src" in message.text and re.search(r"\bwss?://", message.text):
                    gate._unsupported("websocket")
            context.on("console", console_error)
            responses = []
            def collect(response):
                if any(kind in response.headers.get("content-type", "") for kind in ("application/json", "text/csv")):
                    responses.append(response)
            context.on("response", collect)
            response = None
            try:
                response = page.goto(url, wait_until="networkidle", timeout=int(config.get("timeout_ms", 30000)))
                page.wait_for_timeout(int(config.get("render_wait_ms", 300)))
            except Error as exc:
                if not gate.budget_exhausted_kind and not gate.unsupported_contexts and not gate.request_errors:
                    raise
                gate.request_errors.append({"reason": "incomplete_navigation", "error": str(exc)})
            captured = []
            max_bytes = int(config.get("max_bytes", 20_000_000))
            for item in responses:
                try:
                    length = item.headers.get("content-length")
                    if length and int(length) > max_bytes:
                        gate.request_errors.append({"reason": "response_too_large", "url": item.url})
                        continue
                    body = item.body()
                    if len(body) > max_bytes:
                        gate.request_errors.append({"reason": "response_too_large", "url": item.url})
                        continue
                    captured.append({"url": item.url, "status_code": item.status,
                        "content_type": item.headers.get("content-type", ""),
                        "body": base64.b64encode(body).decode()})
                except Exception as exc:
                    gate.request_errors.append({"reason": "response_body_unavailable", "url": item.url, "error": str(exc)})
            try:
                body = page.content().encode()
            except Error:
                body = b""
            if len(body) > max_bytes:
                body = b""
                gate.request_errors.append({"reason": "response_too_large", "url": page.url})
            result = {"body": base64.b64encode(body).decode(), "final_url": page.url,
                      "status_code": response.status if response else 0,
                      "content_type": "text/html", "browser_responses": captured}
            if gate.budget_exhausted_kind:
                result["budget_exhausted_kind"] = gate.budget_exhausted_kind
            if gate.unsupported_contexts:
                result["unsupported_browser_contexts"] = gate.unsupported_contexts
            if gate.request_errors:
                result["request_errors"] = gate.request_errors
            return result
        finally:
            if gate is not None:
                gate.closing = True
            browser.close()
