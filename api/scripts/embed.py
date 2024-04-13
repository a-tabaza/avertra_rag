"""
This script takes a directory of JSON documents and embeds them using the SentenceTransformer model. 
The embeddings are then saved as an NPY file in the specified output folder.
"""

from typing import List, Dict
import sys
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("models/mxbai-embed-large-v1")


def embed(query: str) -> np.array:
    """
    A function that embeds the input query using a pre-trained model.

    Parameters:
        query (str): The input query to be embedded.

    Returns:
        np.array: An array representing the embedded query.

    Raises:
        TypeError: If the input query is not a string.
        ValueError: If the input query is an empty string or contains only whitespace.
        RuntimeError: If the input query cannot be embedded due to an internal error in the model.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string, but got {}".format(type(query)))
    query = query.strip()
    if not query:
        raise ValueError("query cannot be an empty string or contain only whitespace")
    try:
        return model.encode([query]).squeeze(0)
    except RuntimeError as e:
        raise RuntimeError("unable to embed query: {}".format(e)) from e


def get_chunks(chunks_path: str) -> List[Dict]:
    """
    A function that reads and loads JSON data from a specified file path and returns the loaded chunks.

    Parameters:
        chunks_path (str): The file path to the JSON file containing the chunks.

    Returns:
        List[Dict]: A list of dictionaries representing the loaded chunks.

    Raises:
        ValueError: If chunks_path is None.
        FileNotFoundError: If the specified file path does not exist.
        JSONDecodeError: If there is an error decoding the JSON file.
    """

    if chunks_path is None:
        raise ValueError("chunks_path cannot be None")

    try:
        with open(chunks_path, "r") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{chunks_path} does not exist")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(
            None, f"Error decoding JSON file: {chunks_path}", 0, 0, None
        )

    return chunks


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python embed.py <chunks_path> <output_folder> <output_file>")
        exit(1)
    chunks_path = sys.argv[1]
    embeddings = []
    for chunk in tqdm(get_chunks(chunks_path)):
        embeddings.append(embed(chunk["chunk_text"]))
    np.save(f"{sys.argv[2]}/{sys.argv[3]}.npy", np.array(embeddings))
