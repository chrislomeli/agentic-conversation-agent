"""classifier.py — LLM-powered graph nodes for the classification pipeline.

Each ``make_*`` factory returns a node function closed over its LLMClient.
All nodes follow the same contract:
  - Read from JournalState
  - Call the LLM with structured output
  - Return a partial state dict (or Status.ERROR on failure)

Pipeline order (end-of-session):
    exchange_decomposer   — splits transcript into ThreadSegments
    thread_classifier     — tags each thread via taxonomy
    thread_fragment_extractor — distills threads into searchable Fragments

Per-turn:
    intent_classifier     — scores the latest messages to pick a response strategy
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Coroutine, Any

from langchain_core.messages import SystemMessage, HumanMessage

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure.context_builder import ContextBuilder
from journal_agent.configure.prompts import get_prompt
from journal_agent.configure.prompts.helpers import taxonomy_json
from journal_agent.configure.score_card import resolve_scorecard_to_specification
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import (
    JournalState, )
from journal_agent.model.session import Status, UserProfile, PromptKey
from journal_agent.model.session import ThreadSegmentList, ExchangeClassificationRequest, ThreadSegment, Exchange, \
    ThreadClassificationResponse, ExpandedThreadSegment, Fragment, \
    FragmentDraftList, ScoreCard, ContextSpecification
from journal_agent.repository import UserProfileRepository

logger = logging.getLogger(__name__)

# Maximum concurrent LLM calls when fan-out processing threads.
# Prevents rate-limit saturation while still parallelizing.
DEFAULT_LLM_CONCURRENCY = 4


def inflate_threads(threads: list[ThreadSegment], exchanges: list[Exchange]) -> list[ExpandedThreadSegment]:
    """Expand compact ThreadSegments into full dialog text for LLM consumption.

    Each ThreadSegment only holds exchange_ids. This function joins them with
    the actual Exchange content so the LLM can read the conversation.
    """
    # optional make the exchange list into a map
    exchange_map: dict[str, Exchange] = {exchange.exchange_id: exchange for exchange in exchanges}

    # ClassifiedThreadSegment[]
    expanded_threads: list[ExpandedThreadSegment] = []

    # transfer from list[ThreadSegment]
    for thread in threads:
        thread_requests: list[ExchangeClassificationRequest] = []
        # one ThreadSegment
        for exchange_id in thread.exchange_ids:
            exchange = exchange_map[exchange_id]
            t = ExchangeClassificationRequest(exchange_id=exchange.exchange_id,
                                              timestamp=exchange.timestamp,
                                              dialog=f"Human:\n{exchange.human.content}\nAI:\n{exchange.ai.content}\n\n")
            thread_requests.append(t)

        # sort all exchanges by time
        thread_requests.sort(key=lambda r: r.timestamp)
        classification_request = ExpandedThreadSegment(thread_name=thread.thread_name, exchange_ids=thread.exchange_ids,
                                                       exchanges=thread_requests,
                                                       tags=thread.tags)
        expanded_threads.append(classification_request)

    return expanded_threads


def make_exchange_decomposer(llm: LLMClient) -> Callable[..., dict]:
    """Factory: split the session transcript into topical ThreadSegments."""

    @node_trace("exchange_decomposer")
    def exchange_decomposer(state: JournalState) -> dict:
        try:
            system_message = get_prompt(PromptKey.DECOMPOSER)
            system = SystemMessage(system_message)

            exchanges = state["transcript"]

            human_prompt = "\n\n".join([turn.model_dump_json() for turn in exchanges])
            human = HumanMessage(content=human_prompt)

            structured_llm = llm.structured(ThreadSegmentList)
            thread_list = structured_llm.invoke([system, human])

            return {"threads": thread_list.threads}
        except Exception as e:
            logger.exception("Failed to classify turns")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return exchange_decomposer


def make_thread_classifier(llm: LLMClient, max_concurrency: int = DEFAULT_LLM_CONCURRENCY) -> Callable[...,  Coroutine[Any, Any, dict]]:
    """Factory: assign taxonomy tags to each thread via structured LLM output.

    Per-thread LLM calls are fanned out with ``asyncio.gather``, bounded
    by a semaphore to avoid rate-limit saturation.
    """

    @node_trace("thread_classifier")
    async def thread_classifier(state: JournalState) -> dict:

        try:
            system_message = get_prompt(PromptKey.THREAD_CLASSIFIER) + "\n\nTaxonomy:\n" + taxonomy_json()
            system = SystemMessage(system_message)

            exchanges: list[Exchange] = state["transcript"]
            threads: list[ThreadSegment] = state["threads"]
            expanded_threads = inflate_threads(threads, exchanges)

            structured_llm = llm.astructured(ThreadClassificationResponse)
            sem = asyncio.Semaphore(max_concurrency)

            async def classify_one(thread: ExpandedThreadSegment) -> ThreadSegment:
                async with sem:
                    human = HumanMessage(content=thread.model_dump_json())
                    result = await structured_llm.ainvoke([system, human])
                    return ThreadSegment(
                        thread_name=thread.thread_name,
                        exchange_ids=thread.exchange_ids,
                        tags=result.tags,
                    )

            classified_threads = await asyncio.gather(
                *(classify_one(t) for t in expanded_threads)
            )

            return {"classified_threads": list(classified_threads)}

        except Exception as e:
            logger.exception("Failed to classify threads")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return thread_classifier


def make_thread_fragment_extractor(llm: LLMClient, max_concurrency: int = DEFAULT_LLM_CONCURRENCY) -> Callable[
    ...,  Coroutine[Any, Any, dict]]:
    """Factory: distill classified threads into standalone, searchable Fragments.

    Per-thread LLM calls are fanned out with ``asyncio.gather``, bounded
    by a semaphore to avoid rate-limit saturation.
    """

    @node_trace("fragment_extractor")
    async def fragment_extractor(state: JournalState) -> dict:
        try:
            system = SystemMessage(get_prompt(PromptKey.FRAGMENT_EXTRACTOR))

            session_id: str = state["session_id"]
            exchanges: list[Exchange] = state["transcript"]
            threads: list[ThreadSegment] = state["classified_threads"]
            expanded_threads = inflate_threads(threads, exchanges)

            structured_llm = llm.astructured(FragmentDraftList)
            sem = asyncio.Semaphore(max_concurrency)

            async def extract_one(thread: ExpandedThreadSegment) -> list[Fragment]:
                async with sem:
                    human = HumanMessage(content=thread.model_dump_json())
                    # LLM emits only reasoning decisions (content, exchange_ids, tags).
                    # Bookkeeping fields are filled in here, not by the model.
                    result = await structured_llm.ainvoke([system, human])
                    return [
                        Fragment(
                            content=draft.content,
                            exchange_ids=draft.exchange_ids,
                            tags=draft.tags,
                            session_id=session_id,
                            timestamp=datetime.now(),
                        )
                        for draft in result.fragments
                    ]

            nested = await asyncio.gather(
                *(extract_one(t) for t in expanded_threads)
            )
            # flatten list-of-lists into a single fragment list
            fragments = [f for batch in nested for f in batch]

            return {"fragments": fragments}

        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return fragment_extractor


def make_intent_classifier(llm: LLMClient, context_builder: ContextBuilder | None = None) -> Callable[..., dict]:
    """Factory: score recent messages to determine the user's conversational intent.

    Returns a ContextSpecification that drives prompt selection and retrieval
    depth for the current turn.
    """
    context_builder = context_builder or ContextBuilder()

    @node_trace("intent_classifier")
    def intent_classifier(state: JournalState) -> dict:
        try:
            # preconditions
            if len(state["session_messages"]) < 1:
                raise ValueError("No session messages found while asking for AI response")

            # set up the messages we will pass the llm for this intent classification
            intent_spec = ContextSpecification(
                prompt_key=PromptKey.INTENT_CLASSIFIER,
                last_k_session_messages=5,
                last_k_recent_messages=0,
                top_k_retrieved_history=0,
            )
            prompt = get_prompt(key=PromptKey.INTENT_CLASSIFIER, state=state)
            messages = context_builder.get_context(
                prompt=prompt,
                instruction=intent_spec,
                session_messages=state["session_messages"],
            )

            # involve the llm
            structured_llm = llm.structured(ScoreCard)
            score_card: ScoreCard = structured_llm.invoke(messages)

            # handle user profile changes
            if score_card.personalization_score >= 0.5:
                state["user_profile"].is_current = False
                state["user_profile"].is_updated = False

            # translate the score_card into a message specification
            specification = resolve_scorecard_to_specification(score_card)

            # update the state
            return {"context_specification": specification}


        except Exception as e:
            logger.exception("Failed to classify turns")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return intent_classifier


def make_profile_scanner(llm: LLMClient, profile_store: UserProfileRepository, context_builder: ContextBuilder | None = None) -> \
Callable[..., dict]:
    context_builder = context_builder or ContextBuilder()

    @node_trace("profile_scanner")
    def profile_scanner(state: JournalState) -> dict:
        try:
            if state["user_profile"].is_current:
                return {}

            # preconditions
            if len(state["session_messages"]) < 1:
                raise ValueError("No session messages found while asking for AI response")

            # The profile-scanner prompt is parametric: it needs the current
            # UserProfile rendered into its template.
            profile_spec = ContextSpecification(
                prompt_key=PromptKey.PROFILE_SCANNER,
                last_k_session_messages=1,
                last_k_recent_messages=0,
                top_k_retrieved_history=0,
            )
            prompt = get_prompt(key=PromptKey.PROFILE_SCANNER, state=state)
            messages = context_builder.get_context(
                prompt=prompt,
                instruction=profile_spec,
                session_messages=state["session_messages"],
            )

            # involve the llm
            structured_llm = llm.structured(UserProfile)
            user_profile: UserProfile = structured_llm.invoke(messages)

            # at this point we might have changed something about the user profile
            if user_profile.is_updated:
                profile_store.save_profile(user_profile)
                return {"user_profile": user_profile}

            return {}

        except Exception as e:
            logger.exception("Failed to classify turns")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return profile_scanner
