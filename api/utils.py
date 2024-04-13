from pydantic import BaseModel, Field
from typing import Any, Dict, Type


def to_openai_tool(pydantic_class: Type[BaseModel]) -> Dict[str, Any]:
    """Convert pydantic class to OpenAI tool."""
    schema = pydantic_class.model_json_schema()
    function = {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": pydantic_class.model_json_schema(),
    }
    return {"type": "function", "function": function}


def transform_query(query: str) -> str:
    return f"Represent this sentence for searching relevant passages: {query}"
