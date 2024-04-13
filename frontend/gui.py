import os
import sys

from taipy.gui import Gui, State, notify
import openai

from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, Type

import requests
import json

MODEL = "gpt-3.5-turbo-0125"
api_key = ""  # add yours here

client = openai.Client(api_key=api_key)
context = "The following is a conversation with an AI assistant made by Avertra. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by Avertra. How can I help you today? "
conversation = {
    "Conversation": ["What is Avertra all about?", "Simplicity, Synergy, Innovation."]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]


def on_init(state: State) -> None:
    """
    Initialize the app.

    Args:
        - state: The current state of the app.
    """
    state.context = "The following is a conversation with an AI assistant made by Avertra. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by Avertra. How can I help you today? "
    state.conversation = {
        "Conversation": [
            "What is Avertra all about?",
            "Simplicity, Synergy, Innovation.",
        ]
    }
    state.current_user_message = ""
    state.past_conversations = []
    state.selected_conv = None
    state.selected_row = [1]


def to_openai_tool(state: State, pydantic_class: Type[BaseModel]) -> Dict[str, Any]:
    """Convert pydantic class to OpenAI tool."""
    schema = pydantic_class.model_json_schema()
    function = {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": pydantic_class.model_json_schema(),
    }
    return {"type": "function", "function": function}


class QueryType(Enum):
    RETRIEVAL = "1"
    NO_RETRIEVAL = "0"


class Query(BaseModel):
    """
    The user query.
    """

    type: QueryType = Field(
        ...,
        description="The label of the query. Binary flag of zero (if the user query doesn't require retrieval) or one (if the user query requires retrieval).",
    )


def query_expansion(state: State, prompt: str) -> str:
    p_ = f"""
        You have access to chats with queries that are incoming from a user. 
        Your job is to extract the search queries that would help answer the user's query. 
        You will use the search query to search the knowledge base to answer the user's query.
        Ideally, you need to picture yourself being able to do a google search, what would you type to answer the user's query?
        Examples:
        User Query: what are some new energy regulations? 
        Search Query: energy regulation
        User Query: how to save energy during winter?
        Search Query: winter energy conservation
        User Query: 
        {prompt}
        Search Query:"""
    response = (
        client.completions.create(
            model="gpt-3.5-turbo-instruct", prompt=p_, max_tokens=128
        )
        .choices[0]
        .text
    )

    return response


def classifier(state: State, prompt: str) -> str:

    tools = [to_openai_tool(state, Query)]
    p_ = f"""
        You have access to chats with queries that are incoming from a user. 
        Your job is to determine whether the user query requires retrieval or not.
        Retrieval means that the user query is relevant to the knowledge base, and thus you need extra search queries to answer the user's query.
        No Retrieval means that the user query is not relevant to the knowledge base, and thus you don't need extra search queries to answer the user's query. 

        Examples of topics that require retrieval:
        Electricity billing
        Energy consumption measurement
        Energy efficiency programs
        Energy conservation
        Power outage reporting
        Electricity restoration after outage
        Scheduled maintenance outages
        Reducing energy consumption during peak hours
        Energy saving tips for winter
        Energy usage patterns analysis
        Renewable energy programs for consumers
        Solar energy program enrollment
        Benefits of renewable energy sources
        Gas leak safety procedures
        Power outage preparedness
        Portable generator safety guidelines
        Energy regulations
        Environmental standards compliance in the energy sector
        Updates on energy policies
        Energy market trends

        If the query relates to any of the topics above, return 1. If the query doesn't relate to any of the topics above, return 0.
        """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": p_,
            },
            {
                "role": "user",
                "content": f"User Query: {prompt}",
            },
        ],
        tools=tools,
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if tool_calls is None:

        return Query(type=QueryType.NO_RETRIEVAL).model_dump_json()

    outputs = []
    for tool_call in tool_calls:
        function_call = tool_call.function
        # validations to get passed mypy
        assert function_call is not None
        assert function_call.name is not None
        assert function_call.arguments is not None

        name = function_call.name
        arguments_str = function_call.arguments

        if isinstance(function_call.arguments, dict):
            output = Query.model_validate(function_call.arguments)
        else:
            output = Query.model_validate_json(function_call.arguments)

        outputs.append(output.model_dump_json())

        return outputs[0]


def search_and_retrieve(state: State, query: str) -> str:
    results = requests.post(
        url="http://localhost:8000/retrieve/", json={"query": query}
    )
    if results.status_code != 200:
        return None
    return results.json()["results"]


def request(state: State, prompt: str) -> str:
    """
    Send a prompt to the GPT-4 API and return the response.

    Args:
        - state: The current state of the app.
        - prompt: The prompt to send to the API.

    Returns:
        The response from the API.
    """

    label = classifier(state, prompt)
    print(label)

    if json.loads(label)["type"] == QueryType.RETRIEVAL.value:
        search_query = query_expansion(state, prompt)
        print("search query: ", search_query)
        results = search_and_retrieve(state, search_query)
        print("results: ", results)
        if results is not None:
            results = "\n".join(results)
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant created by Avertra, you have access to their knowledge base about energy utility. Use the following context to answer user queries. Answer only from the knowledge base. {results}",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}",
                    },
                ],
                model=MODEL,
            )
            return response.choices[0].message.content

    if json.loads(label)["type"] == QueryType.NO_RETRIEVAL.value:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant created by Avertra, an energy utility company.",
                },
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
            ],
            model=MODEL,
        )
        return response.choices[0].message.content


def update_context(state: State) -> None:
    """
    Update the context with the user's message and the AI's response.

    Args:
        - state: The current state of the app.
    """
    state.context += f"Human: \n {state.current_user_message}\n\n AI:"
    answer = request(state, state.context).replace("\n", "")
    state.context += answer
    state.selected_row = [len(state.conversation["Conversation"]) + 1]
    return answer


def send_message(state: State) -> None:
    """
    Send the user's message to the API and update the context.

    Args:
        - state: The current state of the app.
    """
    notify(state, "info", "Sending message...")
    answer = update_context(state)
    conv = state.conversation._dict.copy()
    conv["Conversation"] += [state.current_user_message, answer]
    state.current_user_message = ""
    state.conversation = conv
    notify(state, "success", "Response received!")


def style_conv(state: State, idx: int, row: int) -> str:
    """
    Apply a style to the conversation table depending on the message's author.

    Args:
        - state: The current state of the app.
        - idx: The index of the message in the table.
        - row: The row of the message in the table.

    Returns:
        The style to apply to the message.
    """
    if idx is None:
        return None
    elif idx % 2 == 0:
        return "user_message"
    else:
        return "gpt_message"


def on_exception(state, function_name: str, ex: Exception) -> None:
    """
    Catches exceptions and notifies user in Taipy GUI

    Args:
        state (State): Taipy GUI state
        function_name (str): Name of function where exception occured
        ex (Exception): Exception
    """
    print(f"An error occured in {function_name}: {ex}")
    notify(state, "error", f"An error occured in {function_name}: {ex}")


def reset_chat(state: State) -> None:
    """
    Reset the chat by clearing the conversation.

    Args:
        - state: The current state of the app.
    """
    state.past_conversations = state.past_conversations + [
        [len(state.past_conversations), state.conversation]
    ]
    state.conversation = {
        "Conversation": ["Who are you?", "Hi! I am GPT-4. How can I help you today?"]
    }


def tree_adapter(item: list) -> [str, str]:
    """
    Converts element of past_conversations to id and displayed string

    Args:
        item: element of past_conversations

    Returns:
        id and displayed string
    """
    identifier = item[0]
    if len(item[1]["Conversation"]) > 3:
        return (identifier, item[1]["Conversation"][2][:50] + "...")
    return (item[0], "Empty conversation")


def select_conv(state: State, var_name: str, value) -> None:
    """
    Selects conversation from past_conversations

    Args:
        state: The current state of the app.
        var_name: "selected_conv"
        value: [[id, conversation]]
    """
    state.conversation = state.past_conversations[value[0][0]][1]
    state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today? "
    for i in range(2, len(state.conversation["Conversation"]), 2):
        state.context += f"Human: \n {state.conversation['Conversation'][i]}\n\n AI:"
        state.context += state.conversation["Conversation"][i + 1]
    state.selected_row = [len(state.conversation["Conversation"]) + 1]


past_prompts = []

page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# Avertra **Chat**{: .color-primary} # {: .logo-text}
<|New Conversation|button|class_name=fullwidth plain|id=reset_app_button|on_action=reset_chat|>
### Previous activities ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|rebuild|>
<|part|class_name=card mt1|
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>
|>
"""

if __name__ == "__main__":
    # if "OPENAI_API_KEY" in os.environ:
    #     api_key = os.environ["OPENAI_API_KEY"]
    # elif len(sys.argv) > 1:
    #     api_key = sys.argv[1]
    # else:
    #     raise ValueError(
    #         "Please provide the OpenAI API key as an environment variable OPENAI_API_KEY or as a command line argument."
    #     )

    Gui(page).run(
        host="0.0.0.0",
        port=8080,
        debug=True,
        dark_mode=True,
        use_reloader=True,
        title="💬 Avertra Chat",
    )
