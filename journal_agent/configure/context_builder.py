"""context_builder.py — Assemble the LLM message list from state and instructions.

ContextBuilder is the single place that decides what the LLM sees on each turn.
Given a ContextSpecification (produced by the intent classifier), it:

1. Truncates session messages, recent messages, and retrieved fragments
   to the counts specified in the instruction.
2. Estimates token usage (fast heuristic, or tiktoken if a model name is given).
3. Prunes in a fixed priority order if the context exceeds the budget:
      retrieved context → recent messages → session messages
4. Wraps the system prompt and optional retrieved context into a SystemMessage.
5. Returns [SystemMessage, ...recent, ...session] ready to pass to the LLM.

Raises ContextBuildError (or subclass) if a required prompt is missing or
the context is still too large after all pruning.
"""

import json
import logging

import tiktoken
from langchain_core.messages import BaseMessage, SystemMessage

from journal_agent.configure.prompts import get_prompt, PromptKey
from journal_agent.model.session import Fragment, ContextSpecification

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ContextBuildError(Exception):
    """Base exception for context-building failures."""


class MissingStateError(ContextBuildError):
    """A required state key or prompt is missing."""

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Required state key missing: {key}")


class ContextTooLargeError(ContextBuildError):
    """Context still exceeds the token budget after all pruning."""

    def __init__(self, tokens: int, budget: int):
        self.tokens = tokens
        self.budget = budget
        super().__init__(
            f"Context is {tokens} tokens, still exceeds budget of {budget} after full pruning"
        )


class ContextBuilder:
    """Assemble and budget-fit the message list sent to the LLM.

    Class-level tunables:
        threshold  — safety margin (0.2 = keep 20% headroom for the reply)
        max_tokens — hard ceiling on estimated context tokens
    """
    threshold: float = 0.2
    per_message_token_overhead: int = 5
    max_tokens: int = 8000
    chars_per_token: int = 4

    def __init__(self):
        pass

    def get_context(self, instruction: ContextSpecification, session_messages: list[BaseMessage] | None = None,
                    recent_messages: list[BaseMessage] | None = None,
                    retrieved_fragments: list[Fragment] | None = None) -> list[BaseMessage]:
        """Build the full message list for one LLM call.

        Returns [SystemMessage, ...recent_messages, ...session_messages].
        Raises MissingStateError if the prompt key is invalid.
        Raises ContextTooLargeError if pruning cannot bring tokens under budget.
        """

        # get all the various components we need to build the context - truncate them based on the instructions passed in
        session_messages = list(
            session_messages[-instruction.last_k_session_messages:]      # truncate to the last N messages based on the instruction passed in
        ) if session_messages and instruction.last_k_session_messages else []  # guard against zeros and Nones

        recent_messages = list(
            recent_messages[-instruction.last_k_recent_messages:]  # truncate to the last N messages based on the instruction passed in
        ) if recent_messages and instruction.last_k_recent_messages else [] # guard against zeros and Nones

        retrieved_fragments = list(
            retrieved_fragments[:instruction.top_k_retrieved_history]   # truncate to the TOP N messages based on the instruction passed in
        ) if retrieved_fragments and instruction.top_k_retrieved_history else []


        # get the prompt value using the PromptKey passed in
        try:
            prompt: str = get_prompt(instruction.prompt_key)
        except KeyError as e:
            raise MissingStateError(f"prompt:{instruction.prompt_key}") from e

        # build the retrieved context to a json string
        retrieved_context: str = json.dumps(
            [{"content": f.content, "tag": [t.tag for t in f.tags]} for f in retrieved_fragments])

        # perform a calculation of the token count
        effective_max = self.max_tokens * (1 - self.threshold)
        count_prompt_tokens = self.count_string_tokens(prompt)
        count_retrieved_context = self.count_string_tokens(retrieved_context)
        count_recent_tokens = self.count_message_tokens(recent_messages)
        count_session_tokens = self.count_message_tokens(session_messages)
        count_all_tokens = count_prompt_tokens + count_retrieved_context + count_recent_tokens + count_session_tokens
        overage_tokens = count_all_tokens - effective_max

        # if we are over - try removing all the retrieved context first
        if overage_tokens > 0:
            # truncate retrieved context
            retrieved_context = ""
            count_all_tokens -= count_retrieved_context
            overage_tokens = max(count_all_tokens - effective_max, 0)
            logger.debug(f"Removing retrieved context reduced overage tokes to {overage_tokens}")

        # truncate recent messages next
        if overage_tokens > 0:
            if count_recent_tokens < overage_tokens:
                recent_messages = []
                count_all_tokens -= count_recent_tokens
                overage_tokens = max(count_all_tokens - effective_max, 0)
                logger.debug(f"Removing recent_messages reduced overage tokes to {overage_tokens}")
            else:
                while overage_tokens > 0 and recent_messages:
                    removed = recent_messages.pop(0)
                    overage_tokens -= self.count_message_tokens([removed])

        if overage_tokens > 0:
            while overage_tokens > 0 and session_messages:
                removed = session_messages.pop(0)
                overage_tokens -= self.count_message_tokens([removed])

            logger.debug(f"Removing session_messages reduced overage tokens to {overage_tokens}")

        # if we've pruned everything we can and are still over, bail
        if overage_tokens > 0:
            logger.debug(
                f"After removing session_messages we are still {overage_tokens} tokens too big - throw an exception")
            raise ContextTooLargeError(int(count_all_tokens), int(effective_max))

        # Construct the System message
        system_content = f"<instructions>\n{prompt}\n</instructions>"
        logger.debug(f"System context: \n{system_content}")

        if retrieved_context and retrieved_fragments:
            rc = f"\n\n<retrieved_context>\n{retrieved_context}\n</retrieved_context>"
            logger.debug(f"Retrieved Context: \n{rc}")
            system_content += rc

        system_message = SystemMessage(
            content=system_content
        )

        # log everything
        logger.debug(
            "Recent Messages: %s",
            [m.model_dump_json() for m in recent_messages]
        )
        logger.debug(
            "Session Messages: %s",
            [m.model_dump_json() for m in session_messages]
        )

        # Construct all messages
        messages = [system_message] + recent_messages + session_messages
        return messages

    def count_message_tokens(self, messages: list, model: str | None = None) -> int:
        """Count tokens: tiktoken if *model* is given, else fast char-based estimate."""
        if isinstance(model, str):
            try:
                return self._count_message_tokens_with_tiktoken(messages, model)
            except Exception:
                pass

        return self._estimate_message_tokens(messages)

    def count_string_tokens(self, content: str, model: str | None = None) -> int:
        """Count tokens for a raw string: tiktoken if *model* is given, else estimate."""
        if isinstance(model, str):
            try:
                enc = tiktoken.encoding_for_model(model)
                return len(enc.encode(content))
            except Exception:
                pass
        return self._estimate_tokens_from_string(content)

    def _count_message_tokens_with_tiktoken(self, messages: list, model: str | None = None) -> int:
        enc = tiktoken.encoding_for_model(model)
        total = 0
        for message in messages:
            total += self.per_message_token_overhead  # every message has overhead tokens
            total += len(enc.encode(message.content))
        total += 2  # reply priming tokens
        return total

    def _estimate_message_tokens(self, messages: list[BaseMessage]) -> int:
        content_chars = sum(len(m.content) for m in messages)  # count
        per_message_overhead = len(
            messages) * self.per_message_token_overhead  # overhead multiplier = 5 in tokens not the same as threshold - this just adds 5 tokens to every message
        return (content_chars // self.chars_per_token) + per_message_overhead

    def _estimate_tokens_from_string(self, content: str) -> int:
        content_chars = len(content)
        return content_chars // self.chars_per_token
