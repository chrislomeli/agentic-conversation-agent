import logging
from collections.abc import Callable

from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import deprecated

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure.prompts import get_prompt, taxonomy_json
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import (
    STATUS_ERROR,
    JournalState, )
from journal_agent.model.session import ClassifiedExchangeList, FragmentList, \
    ThreadSegmentList, ExchangeClassificationRequest, ThreadSegment, Exchange, \
    ThreadClassificationResponse, ExpandedThreadSegment, Fragment

logger = logging.getLogger(__name__)


def inflate_threads(threads: list[ThreadSegment], exchanges: list[Exchange]) -> list[ExpandedThreadSegment]:
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
    @node_trace("exchange_decomposer")
    def exchange_decomposer(state: JournalState) -> dict:
        try:
            system_message = get_prompt("decomposer")
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
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return exchange_decomposer


def make_thread_classifier(llm: LLMClient) -> Callable[..., dict]:
    @node_trace("thread_classifier")
    def thread_classifier(state: JournalState) -> dict:

        try:
            system_message = get_prompt("thread_classifier") + "\n\nTaxonomy:\n" + taxonomy_json()
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
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return thread_classifier


def make_thread_fragment_extractor(llm: LLMClient) -> Callable[..., dict]:
    @node_trace("fragment_extractor")
    def fragment_extractor(state: JournalState) -> dict:
        try:
            system = SystemMessage(get_prompt("extractor"))
            fragment_list: list[Fragment] = []

            exchanges: list[Exchange] = state["transcript"]
            threads: list[ThreadSegment] = state["classified_threads"]
            expanded_threads = inflate_threads(threads, exchanges)

            for expanded_thread in expanded_threads:
                human_prompt = expanded_thread.model_dump_json()
                human = HumanMessage(content=human_prompt)

                structured_llm = llm.structured(Fragment)
                fragment = structured_llm.invoke([system, human])
                fragment_list.append(fragment)

            return {"fragments": fragment_list}

        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return fragment_extractor



def make_exchange_classifier(llm: LLMClient) -> Callable[..., dict]:
    @node_trace("exchange_classifier")
    def exchange_classifier(state: JournalState) -> dict:
        try:
            system_message = get_prompt("classifier") + "\n\nTaxonomy:\n" + taxonomy_json()
            system = SystemMessage(system_message)

            exchanges = state["transcript"]

            human_prompt = "\n\n".join([turn.model_dump_json() for turn in exchanges])
            human = HumanMessage(content=human_prompt)

            structured_llm = llm.structured(ClassifiedExchangeList)
            exchange_list = structured_llm.invoke([system, human])

            return {"classified_exchanges": exchange_list.exchanges}
        except Exception as e:
            logger.exception("Failed to classify turns")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return exchange_classifier



def make_fragment_extractor(llm: LLMClient) -> Callable[..., dict]:
    @node_trace("fragment_extractor")
    def fragment_extractor(state: JournalState) -> dict:
        try:
            system = SystemMessage(get_prompt("extractor"))

            classified = state["classified_exchanges"]
            human_prompt = "\n\n".join([ce.model_dump_json() for ce in classified])
            human = HumanMessage(content=human_prompt)

            structured_llm = llm.structured(FragmentList)
            fragment_list = structured_llm.invoke([system, human])
            #
            # for f in fragments:
            #     f.fragment_id = str(uuid.uuid4())

            return {"fragments": fragment_list.exchanges}
        except Exception as e:
            logger.exception("Failed to extract fragments")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return fragment_extractor
