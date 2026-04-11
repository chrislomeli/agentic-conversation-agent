from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from config_builder import configure_environment
from llm_client import create_llm_client

JOURNAL_SYSTEM_PROMPT = "You are a transcriber who classifies the content of our conversation into one of the following categories: astronomy, biology, chemistry, physics, or other.  Always provide the answer to the question and a classification for the question"



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def main():

    messages: list[BaseMessage] = [SystemMessage(JOURNAL_SYSTEM_PROMPT)]
    settings = configure_environment()
    model_config = settings.selected_model
    client = create_llm_client(model_config.provider, model_config.api_key, model_config.model)
    while True:
        user_input = input("You: ")
        if user_input == "/quit":
            break

        # Add user message
        messages.append(HumanMessage(content=user_input))
        response = client.chat(messages)

        print(response.content)



# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
