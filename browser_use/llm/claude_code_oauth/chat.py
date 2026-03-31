"""
ChatClaudeCodeOAuth — Anthropic LLM via Claude Code / pi OAuth tokens.

Uses the existing ChatAnthropic class but injects OAuth credentials
and required headers instead of a standard API key.

Key requirements for OAuth tokens (sk-ant-oat*):
  1. Pass token as api_key (NOT auth_token)
  2. Beta headers: claude-code-20250219,oauth-2025-04-20
  3. Claude Code identity system prompt (REQUIRED — 403 without it)
  4. user-agent + x-app headers
"""

import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from browser_use.llm.anthropic.chat import ChatAnthropic
from browser_use.llm.claude_code_oauth.credentials import get_valid_token, load_tokens
from browser_use.llm.messages import AssistantMessage, BaseMessage, ContentPartTextParam, SystemMessage, UserMessage
from browser_use.llm.views import ChatInvokeCompletion

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

CLAUDE_CODE_VERSION = "2.1.75"
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

OAUTH_BETA_HEADERS = ",".join([
    "claude-code-20250219",
    "oauth-2025-04-20",
    "fine-grained-tool-streaming-2025-05-14",
    "interleaved-thinking-2025-05-14",
])

OAUTH_HEADERS = {
    "anthropic-beta": OAUTH_BETA_HEADERS,
    "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
    "x-app": "cli",
    "anthropic-dangerous-direct-browser-access": "true",
}


@dataclass
class ChatClaudeCodeOAuth(ChatAnthropic):
    """Anthropic chat model using Claude Code OAuth tokens from ~/.pi/agent/auth.json.

    Thinking is disabled by default because browser-use forces tool_choice
    for structured output, and Anthropic API rejects thinking + forced tool_choice.
    """

    model: str = "claude-sonnet-4-6"
    use_thinking: bool = False  # incompatible with forced tool_choice
    _token_loaded: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        # Load OAuth token and configure parent
        token = get_valid_token()
        self.api_key = token
        self.default_headers = {**(self.default_headers or {}), **OAUTH_HEADERS}
        self._token_loaded = True
        logger.info(f"Loaded OAuth tokens, expires in {load_tokens().expires_in_seconds:.0f}s")

    @property
    def provider(self) -> str:
        return "claude-code-oauth"

    @property
    def name(self) -> str:
        return f"claude-code-oauth/{self.model}"

    def _sanitize_for_oauth(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Rewrite messages for OAuth endpoint compatibility.

        The OAuth endpoint (sk-ant-oat* tokens) enforces two hard constraints:
          1. system MUST be exactly CLAUDE_CODE_IDENTITY (no extra chars)
          2. system MUST be a plain string (not an array with cache_control)

        browser-use puts the full agent prompt in SystemMessage(cache=True).
        We extract that content and inject it into the first UserMessage so the
        model still sees it, while system stays as the exact identity string.

        Also strips cache=True from all messages to avoid cache_control blocks.
        """
        # ── Extract agent instructions from SystemMessage ──
        agent_instructions: str | None = None
        remaining: list[BaseMessage] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                content = msg.content
                if isinstance(content, list):
                    content = "\n".join(p.text for p in content if hasattr(p, "text"))
                # Keep anything beyond the identity string
                if content.strip() != CLAUDE_CODE_IDENTITY:
                    stripped = content.replace(CLAUDE_CODE_IDENTITY, "").strip()
                    agent_instructions = stripped if stripped else content
                # Replace with identity-only system message (cache=False → plain string)
                remaining.append(SystemMessage(content=CLAUDE_CODE_IDENTITY, cache=False))
            else:
                copy = msg.model_copy(deep=True)
                copy.cache = False
                remaining.append(copy)

        # If there was no system message at all, add identity
        if not any(isinstance(m, SystemMessage) for m in remaining):
            remaining.insert(0, SystemMessage(content=CLAUDE_CODE_IDENTITY, cache=False))

        # ── Inject agent instructions into first UserMessage ──
        if agent_instructions:
            for i, msg in enumerate(remaining):
                if isinstance(msg, UserMessage):
                    prefix = f"<AGENT_INSTRUCTIONS>\n{agent_instructions}\n</AGENT_INSTRUCTIONS>\n\n"
                    if isinstance(msg.content, str):
                        remaining[i] = UserMessage(
                            content=prefix + msg.content, cache=False,
                        )
                    elif isinstance(msg.content, list):
                        remaining[i] = UserMessage(
                            content=[ContentPartTextParam(text=prefix)] + list(msg.content),
                            cache=False,
                        )
                    break
            else:
                # No UserMessage found — add one with just the instructions
                remaining.append(UserMessage(content=agent_instructions, cache=False))

        return remaining

    def _refresh_and_retry(self):
        """Force token refresh and update api_key."""
        from browser_use.llm.claude_code_oauth.credentials import refresh_tokens, load_tokens as _load
        tokens = _load()
        new_tokens = refresh_tokens(tokens)
        self.api_key = new_tokens.access

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any
    ) -> ChatInvokeCompletion[T]: ...

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        messages = self._sanitize_for_oauth(messages)
        try:
            return await super().ainvoke(messages, output_format, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if "auth" in err_str or "401" in err_str or "403" in err_str:
                logger.warning("Auth error, forcing token refresh and retrying...")
                self._refresh_and_retry()
                return await super().ainvoke(messages, output_format, **kwargs)
            raise
