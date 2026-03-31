"""
Load and refresh Anthropic OAuth tokens from pi's auth.json.

Token source: ~/.pi/agent/auth.json → anthropic.access / anthropic.refresh
Refresh endpoint: https://console.anthropic.com/v1/oauth/token
Client ID: 9d1c250a-e61b-44d9-88ed-5944d1962f5e (Claude Code PKCE)
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

AUTH_FILE = Path.home() / ".pi" / "agent" / "auth.json"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"


@dataclass
class OAuthTokens:
    access: str
    refresh: str
    expires: float  # ms timestamp

    @property
    def is_expired(self) -> bool:
        return time.time() * 1000 >= self.expires

    @property
    def expires_in_seconds(self) -> float:
        return max(0, (self.expires - time.time() * 1000) / 1000)


def load_tokens() -> OAuthTokens:
    """Load OAuth tokens from pi's auth.json."""
    if not AUTH_FILE.exists():
        raise FileNotFoundError(f"Auth file not found: {AUTH_FILE}")

    with open(AUTH_FILE) as f:
        data = json.load(f)

    anth = data.get("anthropic")
    if not anth or anth.get("type") != "oauth":
        raise ValueError("No Anthropic OAuth credentials in auth.json")

    tokens = OAuthTokens(
        access=anth["access"],
        refresh=anth["refresh"],
        expires=anth["expires"],
    )
    logger.info(f"Loaded Anthropic OAuth tokens from {AUTH_FILE}")
    return tokens


def refresh_tokens(tokens: OAuthTokens) -> OAuthTokens:
    """Refresh expired OAuth tokens and update auth.json."""
    resp = httpx.post(
        TOKEN_URL,
        json={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": tokens.refresh,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    new_tokens = OAuthTokens(
        access=data["access_token"],
        refresh=data.get("refresh_token", tokens.refresh),
        expires=time.time() * 1000 + data["expires_in"] * 1000 - 300_000,  # 5 min buffer
    )

    # Update auth.json
    with open(AUTH_FILE) as f:
        auth_data = json.load(f)
    auth_data["anthropic"]["access"] = new_tokens.access
    auth_data["anthropic"]["refresh"] = new_tokens.refresh
    auth_data["anthropic"]["expires"] = new_tokens.expires
    with open(AUTH_FILE, "w") as f:
        json.dump(auth_data, f, indent=2)

    logger.info(f"Refreshed Anthropic OAuth token, expires in {new_tokens.expires_in_seconds:.0f}s")
    return new_tokens


def get_valid_token() -> str:
    """Load token, refresh if expired, return access token string."""
    tokens = load_tokens()
    if tokens.is_expired:
        logger.info("Token expired, refreshing...")
        tokens = refresh_tokens(tokens)
    return tokens.access
