import json
import logging
from collections.abc import Callable
from dataclasses import asdict

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure.prompts import get_prompt, TAXONOMY
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import (
    STATUS_ERROR,
    JournalState,
)
from journal_agent.model.session import Role, ClassifiedExchange
from journal_agent.storage.api import SessionStore

logger = logging.getLogger(__name__)

def _make_turn_classifier(llm: LLMClient, session_store: SessionStore) -> Callable[..., dict]:
    @node_trace("turn_classifier")
    def turn_classifier(state: JournalState) -> dict:
        try:
            taxonomy_json = json.dumps([asdict(t) for t in TAXONOMY])
            system_message = get_prompt("classifier") + "\n\nTaxonomy:\n" + taxonomy_json
            system = SystemMessage(system_message)

            turns = session_store.get_cached_turns()
            human_prompt = "\n\n".join([turn.model_dump_json() for turn in turns])
            human = HumanMessage(content=human_prompt)

            structured_llm = llm.structured(list[ClassifiedExchange])
            exchanges = structured_llm.invoke([system, human])

            return {"classified_exchanges": exchanges}
        except Exception as e:
            logger.exception("Failed to classify turns")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }