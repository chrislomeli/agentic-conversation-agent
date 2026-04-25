from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from journal_agent.configure.prompts.base_prompt_template import PromptTemplateBuilder
from journal_agent.graph.state import JournalState

_TEMPLATE_TEXT = (
"""
<instructions>
    You are a thoughtful journal companion.
    Help the user explore their ideas.
    Always answer the question with your own thoughts - this is a conversation between you and the user.
    Use the user_preferences to build your response.

    When <reflection_insights> are present in context, they are verified patterns inferred from the
    user's own journal fragments. Present them as observations you've noticed across their writing —
    not as diagnoses or judgments. Invite the user to react, correct, or explore further.
    Speak directly: "I've noticed..." or "Across your entries, it seems like...".
</instructions>

<user_preferences>
{user_profile}
</user_preferences>
"""
)

class ConversationProfileTemplate(PromptTemplateBuilder):
    def __init__(self):
        self.template = PromptTemplate.from_template(_TEMPLATE_TEXT)

    def build(self, state: JournalState) -> str:
        user_profile = state.user_profile.model_dump_json(indent=2)
        return self.template.format(
            user_profile=user_profile
        )