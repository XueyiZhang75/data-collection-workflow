"""Explicit account-level model refusals; ordinary request/rate failures stay local."""
from __future__ import annotations

class ProviderAccountLimit(RuntimeError):
    def __init__(self, provider, state, *, dispatched=False):
        self.provider = provider
        self.state = dict(state)
        self.dispatched = bool(dispatched)
        super().__init__(f"provider account limit ({provider}): {state.get('reason', 'account unavailable')}")

def provider_account_limit(error):
    """Require an SDK HTTP error and explicit account quota/credit semantics."""
    status = getattr(error, 'status_code', None)
    body = getattr(error, 'body', None)
    if status not in {400, 402, 403, 429} or not isinstance(body, dict):
        return None
    detail = body.get('error', body)
    if not isinstance(detail, dict):
        return None
    code = str(detail.get('code') or detail.get('type') or '').lower()
    message = str(detail.get('message') or '')
    normalized = ' '.join(message.casefold().split())
    explicit = code in {'insufficient_quota', 'insufficient_credits', 'credit_balance_too_low', 'billing_hard_limit_reached'}
    explicit = explicit or ('you have reached your specified api usage limits' in normalized and 'regain access' in normalized)
    explicit = explicit or ('credit balance is too low' in normalized)
    if not explicit:
        return None
    return {'reason': message, 'error_code': code, 'http_status': status}
