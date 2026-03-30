"""
Claude Code OAuth credential management.

Reads OAuth tokens from known credential stores (pi agent, Claude Code CLI),
refreshes expired tokens, and provides the access token for API calls.

Token refresh uses the same OAuth endpoint as Claude Code / pi:
  POST https://platform.claude.com/v1/oauth/token
  { grant_type: "refresh_token", client_id: CLIENT_ID, refresh_token: "..." }
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e'
TOKEN_URL = 'https://platform.claude.com/v1/oauth/token'

# Safety margin: refresh 5 minutes before actual expiry
REFRESH_MARGIN_MS = 5 * 60 * 1000


@dataclass
class OAuthTokens:
	"""OAuth token set with refresh capability."""

	access: str
	refresh: str
	expires: int  # ms since epoch

	@property
	def is_expired(self) -> bool:
		return int(time.time() * 1000) >= self.expires

	@property
	def expires_in_seconds(self) -> float:
		return max(0, (self.expires - int(time.time() * 1000)) / 1000)


def _default_credential_paths() -> list[Path]:
	"""Return known credential file paths in priority order."""
	home = Path.home()
	return [
		home / '.pi' / 'agent' / 'auth.json',  # pi coding agent
		home / '.claude' / 'auth.json',  # Claude Code CLI (if it stores here)
	]


def load_tokens(credential_path: str | Path | None = None) -> OAuthTokens:
	"""
	Load Anthropic OAuth tokens from a credential file.

	Args:
		credential_path: Explicit path to auth.json. If None, searches default locations.

	Returns:
		OAuthTokens with access, refresh, and expiry.

	Raises:
		FileNotFoundError: No credential file found.
		ValueError: File found but missing Anthropic OAuth credentials.
	"""
	if credential_path is not None:
		paths = [Path(credential_path)]
	else:
		paths = _default_credential_paths()

	for path in paths:
		if not path.exists():
			continue

		try:
			data = json.loads(path.read_text())
		except (json.JSONDecodeError, OSError) as e:
			logger.warning(f'Failed to read {path}: {e}')
			continue

		anthropic = data.get('anthropic')
		if not anthropic:
			continue

		if anthropic.get('type') != 'oauth':
			continue

		access = anthropic.get('access')
		refresh = anthropic.get('refresh')
		expires = anthropic.get('expires')

		if not all([access, refresh, expires]):
			logger.warning(f'Incomplete OAuth credentials in {path}')
			continue

		logger.info(f'Loaded Anthropic OAuth tokens from {path}')
		return OAuthTokens(access=access, refresh=refresh, expires=int(expires))

	searched = [str(p) for p in paths]
	raise FileNotFoundError(
		f'No Anthropic OAuth credentials found. Searched: {searched}. '
		f'Login with Claude Code CLI or pi (/login) first.'
	)


async def refresh_tokens(tokens: OAuthTokens) -> OAuthTokens:
	"""
	Refresh expired OAuth tokens using the refresh token.

	Args:
		tokens: Current token set with valid refresh token.

	Returns:
		New OAuthTokens with fresh access token.

	Raises:
		httpx.HTTPStatusError: If the refresh request fails.
		ValueError: If the response is malformed.
	"""
	async with httpx.AsyncClient(timeout=30.0) as client:
		response = await client.post(
			TOKEN_URL,
			json={
				'grant_type': 'refresh_token',
				'client_id': CLIENT_ID,
				'refresh_token': tokens.refresh,
			},
			headers={
				'Content-Type': 'application/json',
				'Accept': 'application/json',
			},
		)
		response.raise_for_status()

	data = response.json()

	new_tokens = OAuthTokens(
		access=data['access_token'],
		refresh=data['refresh_token'],
		expires=int(time.time() * 1000) + data['expires_in'] * 1000 - REFRESH_MARGIN_MS,
	)

	logger.info(f'Refreshed Anthropic OAuth token, expires in {new_tokens.expires_in_seconds:.0f}s')
	return new_tokens


def save_tokens(tokens: OAuthTokens, credential_path: str | Path | None = None) -> None:
	"""
	Persist refreshed tokens back to the credential file.

	Updates only the anthropic entry, preserving other provider credentials.

	Args:
		tokens: The refreshed token set.
		credential_path: Path to auth.json. If None, uses first default path.
	"""
	if credential_path is not None:
		path = Path(credential_path)
	else:
		# Save to first default path that exists, or first path if none exist
		paths = _default_credential_paths()
		path = next((p for p in paths if p.exists()), paths[0])

	# Read existing data
	data: dict = {}
	if path.exists():
		try:
			data = json.loads(path.read_text())
		except (json.JSONDecodeError, OSError):
			data = {}

	# Update anthropic entry
	data['anthropic'] = {
		'type': 'oauth',
		'access': tokens.access,
		'refresh': tokens.refresh,
		'expires': tokens.expires,
	}

	# Write back with restricted permissions
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(data, indent=2))
	path.chmod(0o600)

	logger.debug(f'Saved refreshed tokens to {path}')
