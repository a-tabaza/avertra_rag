"""
This script takes a directory of JSON documents and splits them into smaller chunks using Langchain's RecursiveCharacterTextSplitter.
The chunks are then saved as a JSON file in the specified output folder.
"""

import sys
import os
import json
from typing import List, Dict
import glob
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)


def read_documents(document_path: str) -> List[Dict]:
    """
    Read JSON documents from the specified path.

    Args:
        document_path (str): The path to the directory containing the JSON documents.

    Returns:
        List[Dict]: A list of dictionaries representing the loaded JSON documents.

    Raises:
        FileNotFoundError: If document_path does not exist or is not a directory.
        json.JSONDecodeError: If there is an error decoding a JSON file.
    """
    if not os.path.isdir(document_path):
        raise FileNotFoundError(f"{document_path} is not a directory")

    document_files = glob.glob(f"{document_path}/*.json")
    documents = []
    for document_file in document_files:
        try:
            with open(document_file, "r") as f:
                documents.append(json.load(f))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file: {document_file} - {e}")

    if len(documents) == 0:
        print(f"read 0 documents from {document_path}")
    else:
        print(f"read {len(documents)} documents")

    return documents


def chunk(documents: List[Dict]) -> List[List[Dict]]:
    """
    Split the text of each document into chunks and create a list of chunk dictionaries.

    Args:
        documents (List[Dict]): A list of dictionaries representing the documents.

    Returns:
        List[List[Dict]]: A list of lists of dictionaries representing the chunks.

    Raises:
        TypeError: If document is not a list.
        TypeError: If document is not a dictionary.
        KeyError: If document is missing the "text" key.
    """
    if not isinstance(documents, list):
        raise TypeError("documents must be a list")

    full_text = """
    The following is an excerpt of a document titled: {title}
    {text} 
    """
    chunks = []
    for document in documents:
        if not isinstance(document, dict):
            raise TypeError("document must be a dictionary")
        if "text" not in document:
            raise KeyError("document is missing the 'text' key")

        for chunk in text_splitter.split_text(document["text"]):
            chunk_content = {
                "title": document["meta"]["title"],
                "text": chunk,
            }
            chunks.append(
                {
                    "chunk_id": str(uuid4()),
                    "document_id": document["id"],
                    "chunk_text": full_text.format(**chunk_content),
                }
            )
    return chunks


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python chunk.py <document_path> <output_folder> <output_file>")
        exit(1)
    documents = read_documents(sys.argv[1])
    titles = [document["meta"]["title"] for document in documents]
    chunks = chunk(documents)
    json.dump(chunks, open(f"{sys.argv[2]}/{sys.argv[3]}.json", "w"))
    print(f"\nchunked {len(titles)} documents into {len(chunks)} chunks\n")
    print("document titles: ", ", ".join(titles))
