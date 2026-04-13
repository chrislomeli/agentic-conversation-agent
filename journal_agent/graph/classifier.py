import json
import logging
from collections.abc import Callable

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure.prompts import get_prompt, TAXONOMY
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.state import (
    STATUS_ERROR,
    JournalState,
)
from journal_agent.model.session import Role
from journal_agent.storage.api import SessionStore

logger = logging.getLogger(__name__)

def _make_turn_classifier(llm: LLMClient, session_store: SessionStore) -> Callable[..., dict]:
    @node_trace("turn_classifier")
    def turn_classifier(state: JournalState) -> dict:
        try:
            # input = all turns from store
            turns = session_store.get_cached_turns()
            human_prompt = "Please classify these turns using the following Taxonomy : "  + json.dumps(TAXONOMY)
            system = SystemMessage(get_prompt("classifier"))

            human = HumanMessage(content= < serialized turns + taxonomy >)
            result = llm.chat([system, human])

            # system prompt < key == classifier   -
            # human prompt < key == classification task -   `
            # llm client < key == classifier

            messages = state["seed_context"] + state["session_messages"]
            response = llm.chat(messages)  # model answers using context
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            session_store.cache_turn(
                session_id=state["session_id"],
                role=Role.AI,
                content=content,
            )
            print(content)
            return {
                "session_messages": [AIMessage(content=content)],
                "status": ""
            }
        except Exception as e:
            logger.exception("Failed to generate AI response")
            return {
                "status": STATUS_ERROR,
                "error_message": str(e),
            }

    return turn_classifier