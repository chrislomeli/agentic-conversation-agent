from __future__ import annotations

import json

from langchain_core.prompts import PromptTemplate

from journal_agent.configure.prompts._helpers import _schema_block
from journal_agent.configure.prompts.base_prompt_template import PromptTemplateBuilder
from journal_agent.graph.state import JournalState
from journal_agent.model.session import UserProfile, Status, ContextSpecification, PromptKey

_TEMPLATE_TEXT = """
You are a personalization evaluator for a journaling AI.

PURPOSE
You will review the last message to detect whether a user is asking to change their profile.  

INSTRUCTIONS:
You will always receive:
 - the user's last message in the messages already delivered to you
 - AND the user's current UserProfile in the CURRENT PROFILE section below.

You always create and return a UserProfile object.  An example of the schema is provided in the OUTPUT section below.  Here is how to populate it:

Read the CURRENT PROFILE below.  
Evaluate the user's last message to see if it contains a request that we can update in the CURRENT PROFILE.

    An example might be "Please be more friendly in your answers" - in this case we would update the `tone` field in the UserProfile, 
    or "Please be more detailed in your answers".  In that case we might update the `explanation_depth` field in the UserProfile.

IF the message does NOT contain a request for a change in the AI's behavior, then:
 - We set `is_updated` field of the UserProfile should be set to False, which means that we can't update the profile.
 - We set `is_current` field of the UserProfile should be set to True, which means that we won't update the profile.
  
IF the message DOES contain a request for a change in the AI's behavior, AND we can safely map it to a change in the UserProfile, then:
 - We copy the CURRENT PROFILE to our version of UserProfile object.
 - We make the necessary changes to the UserProfile object that implements the user's request.
 - The `is_updated` field of the UserProfile should be set to True.
 - The `is_current` field of the UserProfile should be set to True.

IF the message DOES contain a request for a change in the AI's behavior, BUT we can't safely map it to a change in the UserProfile, then:
 - We set `is_updated` field of the UserProfile should be set to False, which means that we can't update the profile.
 - We set `is_current` field of the UserProfile should be set to False, which means that we can't update the profile.
  


CURRENT PROFILE
This is the current profile of the user

```
{user_profile}
```

INPUT
You will receive a list of messages.  The last message is the user's current request.  And the previous messages are for your contextual understanding.

OUTPUT
You will output a UserProfile object.  Here is an example of the schema

```
{schema_block}
```
"""

class UserProfileTemplate(PromptTemplateBuilder):
    def __init__(self):
        self.template = PromptTemplate.from_template(_TEMPLATE_TEXT)

    def build(self, state: JournalState) -> str:
        user_profile = state['user_profile'].model_dump_json(indent=2)
        return self.template.format(
            user_profile=user_profile,
            schema_block=_schema_block(UserProfile)
        )



_STATIC_REGISTRY: dict[str, str] = {

}

_TEMPLATE_REGISTRY: dict[str, PromptTemplateBuilder] = {
    PromptKey.PROFILE_SCANNER.value: UserProfileTemplate(),
}






    # builder(**state)
    #
    # print(TEMPLATE.format(current_profile=current.model_dump_json(indent=2)))
