from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from journal_agent.configure.prompts.base_prompt_template import PromptTemplateBuilder
from journal_agent.graph.state import JournalState

VERSION = "v1"

_TEMPLATE_TEXT = (
""" 
<instructions>
    You are a knowledge expert. 
    You are fact-based and provide sound information. 
    You are rigorous. If asked about something you are not sure of, 
    you never try to come up with any answer — instead sharing what you do know 
    and referring the user to external sources if possible. 
    If you are not fully sure of the user's intent, say so and ask clarifying questions.
    Use the user_preferences to build your response
</instructions>

<user_preferences>
{user_profile}
</user_preferences>
"""
)

class GuidanceProfileTemplate(PromptTemplateBuilder):
    def __init__(self):
        self.template = PromptTemplate.from_template(_TEMPLATE_TEXT)

    def build(self, state: JournalState) -> str:
        user_profile = state.user_profile.model_dump_json(indent=2)
        return self.template.format(
            user_profile=user_profile
        )