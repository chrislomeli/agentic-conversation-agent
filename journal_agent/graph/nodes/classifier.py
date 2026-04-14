import logging
import uuid
from collections.abc import Callable

from langchain_core.messages import SystemMessage, HumanMessage

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure.prompts import get_prompt, taxonomy_json
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import (
    STATUS_ERROR,
    JournalState, STATUS_FRAGMENTS_SAVED, STATUS_EXCHANGES_SAVED, STATUS_TRANSCRIPT_SAVED,
)
from journal_agent.model.session import ClassifiedExchange, Fragment, ClassifiedExchangeList, FragmentList
from journal_agent.storage.exchange_store import TranscriptStore
from journal_agent.storage.storage import JsonStore

logger = logging.getLogger(__name__)

def make_exchange_classifier(llm: LLMClient, session_store: TranscriptStore) -> Callable[..., dict]:
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

