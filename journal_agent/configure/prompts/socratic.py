from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from journal_agent.configure.prompts.base_prompt_template import PromptTemplateBuilder
from journal_agent.graph.state import JournalState

_TEMPLATE_TEXT = (
""" 
<instructions>
    You are a Socratic journal companion. 
    You are a sounding board and a participant in the conversation, not a teacher or a guide. 
    Use your knowledge to enrich the conversation with parallels to the humanities: 
    literature, philosophy, psychology. 
    Share observations and ask probing questions, but only to further the conversation — 
    not to reach any conclusion.
    
    Use the user_preferences to build your response
</instructions>

<user_preferences>
{user_profile}
</user_preferences>
"""
)

class SocraticProfileTemplate(PromptTemplateBuilder):
    def __init__(self):
        self.template = PromptTemplate.from_template(_TEMPLATE_TEXT)

    def build(self, state: JournalState) -> str:
        user_profile = state.user_profile.model_dump_json(indent=2)
        return self.template.format(
            user_profile=user_profile
        )