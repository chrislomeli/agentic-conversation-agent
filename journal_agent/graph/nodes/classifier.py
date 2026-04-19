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

import logging
from collections.abc import Callable
from datetime import datetime

from langchain_core.messages import SystemMessage, HumanMessage

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure.context_builder import ContextBuilder
from journal_agent.configure.prompts import get_prompt, taxonomy_json, PromptKey
from journal_agent.configure.score_card import resolve_scorecard_to_specification
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import (
    JournalState, )
from journal_agent.model.session import Status, UserProfile
from journal_agent.model.session import ThreadSegmentList, ExchangeClassificationRequest, ThreadSegment, Exchange, \
    ThreadClassificationResponse, ExpandedThreadSegment, Fragment, \
    FragmentDraftList, ScoreCard, ContextSpecification
from journal_agent.storage.profile_store import UserProfileStore

logger = logging.getLogger(__name__)

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


def make_thread_classifier(llm: LLMClient) -> Callable[..., dict]:
    """Factory: assign taxonomy tags to each thread via structured LLM output."""
    @node_trace("thread_classifier")
    def thread_classifier(state: JournalState) -> dict:

        try:
            system_message = get_prompt(PromptKey.THREAD_CLASSIFIER) + "\n\nTaxonomy:\n" + taxonomy_json()
            system = SystemMessage(system_message)

            classified_threads: list[ThreadSegment] = []

            exchanges: list[Exchange] = state["transcript"]
            threads: list[ThreadSegment] = state["threads"]
            expanded_threads = inflate_threads(threads, exchanges)
            for expanded_thread in expanded_threads:
                human_prompt = expanded_thread.model_dump_json()
                human = HumanMessage(content=human_prompt)

                # call the llm for one request
                structured_llm = llm.structured(ThreadClassificationResponse)
                result = structured_llm.invoke([system, human])
                expanded_thread.tags = result.tags
                classified_threads.append(ThreadSegment(
                    thread_name=expanded_thread.thread_name,
                    exchange_ids=expanded_thread.exchange_ids,
                    tags=result.tags
                ))

            return {"classified_threads": classified_threads}

        except Exception as e:
            logger.exception("Failed to classify turns")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return thread_classifier


def make_thread_fragment_extractor(llm: LLMClient) -> Callable[..., dict]:
    """Factory: distill classified threads into standalone, searchable Fragments."""
    @node_trace("fragment_extractor")
    def fragment_extractor(state: JournalState) -> dict:
        try:
            system = SystemMessage(get_prompt(PromptKey.FRAGMENT_EXTRACTOR))
            fragment_list: list[Fragment] = []

            session_id: str = state["session_id"]
            exchanges: list[Exchange] = state["transcript"]
            threads: list[ThreadSegment] = state["classified_threads"]
            expanded_threads = inflate_threads(threads, exchanges)

            structured_llm = llm.structured(FragmentDraftList)

            for expanded_thread in expanded_threads:
                human_prompt = expanded_thread.model_dump_json()
                human = HumanMessage(content=human_prompt)

                # LLM emits only reasoning decisions (content, exchange_ids, tags).
                # Bookkeeping fields are filled in here, not by the model.
                result = structured_llm.invoke([system, human])
                for draft in result.fragments:
                    fragment_list.append(Fragment(
                        content=draft.content,
                        exchange_ids=draft.exchange_ids,
                        tags=draft.tags,
                        session_id=session_id,
                        timestamp=datetime.now(),
                    ))

            return {"fragments": fragment_list}

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
            messages = context_builder.get_context(
                ContextSpecification(
                    prompt_key=PromptKey.INTENT_CLASSIFIER,
                    last_k_session_messages=5,
                    last_k_recent_messages=0,
                    top_k_retrieved_history=0
                ) ,
                session_messages=state["session_messages"])

            # involve the llm
            structured_llm = llm.structured(ScoreCard)
            score_card: ScoreCard = structured_llm.invoke(messages)

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



def make_profile_scanner(llm: LLMClient, context_builder: ContextBuilder | None = None) -> Callable[..., dict]:
    context_builder = context_builder or ContextBuilder()
    @node_trace("retrieve_history")
    def profile_scanner(state: JournalState) -> dict:
        try:
            if state["user_profile"].is_current:
                return {}

            # preconditions
            if len(state["session_messages"]) < 1:
                raise  Exception("No session messages found while asking for AI response")

            # The profile-scanner prompt is parametric: it needs the current
            # UserProfile rendered into its template via prompt_vars. Using a
            # default profile until the JournalState carries one through.
            current_profile = UserProfile()

            messages = context_builder.get_context(
                ContextSpecification(
                    prompt_key=PromptKey.PROFILE_CLASSIFIER,
                    last_k_session_messages=5,
                    last_k_recent_messages=0,
                    top_k_retrieved_history=0,
                    prompt_vars={"current_profile": current_profile.model_dump_json(indent=2)},
                ),
                session_messages=state["session_messages"])

            # involve the llm
            structured_llm = llm.structured(UserProfile)
            user_profile: UserProfile = structured_llm.invoke(messages)

            # at this point we might have changed something about the user profile

            if user_profile.is_current:
                UserProfileStore().save_profile(user_profile)

                # push into state
                return {"user_profile": user_profile}

        except Exception as e:
            logger.exception("Failed to classify turns")
            return {
                "status": Status.ERROR,
                "error_message": str(e),
            }

    return profile_scanner


