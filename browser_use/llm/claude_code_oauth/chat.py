"""
ChatClaudeCodeOAuth — Anthropic chat model using Claude Code OAuth tokens.

Wraps ChatAnthropic with automatic OAuth token loading and refresh.
No API key needed — uses tokens from Claude Code CLI or pi agent.

OAuth tokens require specific headers that differ from standard API key auth:
  - Bearer auth via auth_token (not x-api-key)
  - anthropic-beta: claude-code-20250219,oauth-2025-04-20
  - user-agent: claude-cli/<version>
  - x-app: cli

Usage:
	from browser_use.llm.claude_code_oauth.chat import ChatClaudeCodeOAuth

	model = ChatClaudeCodeOAuth(model='claude-sonnet-4-20250514')
	# or with explicit credential path:
	model = ChatClaudeCodeOAuth(model='claude-sonnet-4-20250514', credential_path='~/.pi/agent/auth.json')
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from browser_use.llm.anthropic.chat import ChatAnthropic
from browser_use.llm.base import BaseChatModel
from browser_use.llm.claude_code_oauth.credentials import (
	OAuthTokens,
	load_tokens,
	refresh_tokens,
	save_tokens,
)
from browser_use.llm.messages import BaseMessage, SystemMessage
from browser_use.llm.views import ChatInvokeCompletion

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# Headers required for OAuth token auth (from pi-ai anthropic provider)
CLAUDE_CODE_VERSION = '2.1.75'
OAUTH_BETA_FEATURES = 'claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14'
OAUTH_DEFAULT_HEADERS = {
	'accept': 'application/json',
	'anthropic-dangerous-direct-browser-access': 'true',
	'anthropic-beta': OAUTH_BETA_FEATURES,
	'user-agent': f'claude-cli/{CLAUDE_CODE_VERSION}',
	'x-app': 'cli',
}
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


@dataclass
class ChatClaudeCodeOAuth(BaseChatModel):
	"""
	Anthropic chat model authenticated via Claude Code OAuth.

	Reads OAuth tokens from credential stores (pi agent auth.json or Claude Code CLI),
	auto-refreshes expired tokens, and delegates to ChatAnthropic.

	Args:
		model: Anthropic model name (e.g. 'claude-sonnet-4-20250514')
		credential_path: Explicit path to auth.json. If None, searches default locations.
		max_tokens: Maximum tokens in response.
		temperature: Sampling temperature.
		persist_refreshed_tokens: Whether to save refreshed tokens back to disk.
	"""

	model: str = 'claude-sonnet-4-20250514'
	credential_path: str | Path | None = None
	max_tokens: int = 8192
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None
	max_retries: int = 10
	persist_refreshed_tokens: bool = True

	# Internal state
	_tokens: OAuthTokens | None = field(default=None, init=False, repr=False)
	_delegate: ChatAnthropic | None = field(default=None, init=False, repr=False)

	@property
	def provider(self) -> str:
		return 'anthropic'

	@property
	def name(self) -> str:
		return str(self.model)

	async def _ensure_tokens(self) -> OAuthTokens:
		"""Load and refresh OAuth tokens as needed."""
		if self._tokens is None:
			self._tokens = load_tokens(self.credential_path)
			logger.info(f'Loaded OAuth tokens, expires in {self._tokens.expires_in_seconds:.0f}s')

		if self._tokens.is_expired:
			logger.info('OAuth token expired, refreshing...')
			self._tokens = await refresh_tokens(self._tokens)
			if self.persist_refreshed_tokens:
				save_tokens(self._tokens, self.credential_path)
			# Invalidate delegate so it gets recreated with new token
			self._delegate = None

		return self._tokens

	def _build_delegate(self, tokens: OAuthTokens) -> ChatAnthropic:
		"""Create a ChatAnthropic instance using the current OAuth access token.

		OAuth tokens (sk-ant-oat01-...) require Bearer auth via auth_token
		plus special beta headers (oauth-2025-04-20, claude-code-20250219).
		This matches how pi and Claude Code CLI authenticate.
		"""
		return ChatAnthropic(
			model=self.model,
			auth_token=tokens.access,
			default_headers=OAUTH_DEFAULT_HEADERS,
			max_tokens=self.max_tokens,
			temperature=self.temperature,
			top_p=self.top_p,
			seed=self.seed,
			max_retries=self.max_retries,
		)

	async def _get_delegate(self) -> ChatAnthropic:
		"""Get or create the delegate ChatAnthropic with a valid token."""
		tokens = await self._ensure_tokens()
		if self._delegate is None:
			self._delegate = self._build_delegate(tokens)
		return self._delegate

	def _inject_identity(self, messages: list[BaseMessage]) -> list[BaseMessage]:
		"""Prepend Claude Code identity to the system prompt.

		OAuth tokens REQUIRE a system prompt starting with the Claude Code identity.
		If a SystemMessage exists, prepend the identity to its content.
		If none exists, insert one at the start.
		"""
		messages = [m.model_copy(deep=True) for m in messages]

		# Find existing SystemMessage
		for i, msg in enumerate(messages):
			if isinstance(msg, SystemMessage):
				existing = msg.content if isinstance(msg.content, str) else ''
				messages[i] = SystemMessage(
					content=f'{CLAUDE_CODE_IDENTITY}\n\n{existing}'.strip(),
					cache=msg.cache,
				)
				return messages

		# No SystemMessage found — insert one
		messages.insert(0, SystemMessage(content=CLAUDE_CODE_IDENTITY))
		return messages

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		delegate = await self._get_delegate()
		messages_with_identity = self._inject_identity(messages)
		try:
			return await delegate.ainvoke(messages_with_identity, output_format=output_format, **kwargs)
		except Exception as e:
			# If auth fails, try one token refresh and retry
			error_str = str(e).lower()
			if 'authentication' in error_str or 'unauthorized' in error_str or '401' in error_str:
				logger.warning('Auth error, forcing token refresh and retrying...')
				if self._tokens is not None:
					self._tokens = await refresh_tokens(self._tokens)
					if self.persist_refreshed_tokens:
						save_tokens(self._tokens, self.credential_path)
					self._delegate = self._build_delegate(self._tokens)
					return await self._delegate.ainvoke(messages_with_identity, output_format=output_format, **kwargs)
			raise
