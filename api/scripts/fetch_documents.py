"""
This takes in a list of seed queries and fetches relevant documents from Wikipedia. It is meant to be used to populate a vector index. 
The script uses the mediawiki and flashrank libraries to search for relevant documents and rank them based on their relevance to the seed queries. 
The script then writes the relevant documents to a specified output folder. 
The script is meant to be used in conjunction with other scripts to create a vector index for a search engine.
"""

import sys
from typing import List, Dict
from mediawiki import MediaWiki
from flashrank import Ranker, RerankRequest
from tqdm import tqdm
import json

wikipedia = MediaWiki()
ranker = Ranker()


def get_seed_queries(seed_queries_path: str) -> List[str]:
    """
    A function that reads a file and returns a list of stripped lines.

    Parameters:
        seed_queries (str): The file path to read.

    Returns:
        List[str]: A list of stripped lines from the file.

    Raises:
        FileNotFoundError: If seed_queries_path does not exist as a file.
        TypeError: If seed_queries_path is not a string.
    """
    if not isinstance(seed_queries_path, str):
        raise TypeError("seed_queries must be a string")

    try:
        with open(seed_queries_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"{seed_queries_path} does not exist")


def search_and_retrieve(query: str) -> List[Dict]:
    """
    A function that searches for a query in Wikipedia, retrieves relevant pages,
    extracts specific information from each page, reranks the results,
    and returns only the results with a score higher than 0.8.

    Parameters:
    query (str): The search query to be executed.

    Returns:
    List[Dict]: A list of dictionaries containing information about the relevant pages.
    """
    try:
        results = wikipedia.search(query)
        pages = []
        for result in results:
            try:
                page = wikipedia.page(result)
                pages.append(page)
            except Exception as e:
                print(f"Error retrieving page {result}: {e}")
        pages_full = [
            {
                "id": page.pageid,
                "text": page.content,
                "meta": {"title": page.title, "summary": page.summarize(chars=256)},
            }
            for page in pages
            if page is not None
        ]
        rerankrequest = RerankRequest(query=query, passages=pages_full)
        results_ = ranker.rerank(rerankrequest)
        k = 3 if len(results_) > 3 else len(results_)
        results_ = [result for result in results_[:k]]
        return results_
    except Exception as e:
        print(f"search_and_retrieve raised exception: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_documents.py <seed_queries_file> <output_folder>")
        exit(1)

    seed_queries = get_seed_queries(sys.argv[1])
    documents = []
    for query in tqdm(seed_queries):
        print("\nQuery: ", query)
        results = search_and_retrieve(query)
        documents.extend(results)

    for document in documents:
        print("writing document: ", document["meta"]["title"])
        file_name = f"{document['meta']['title'].strip().lower().replace(' ', '_')}"
        del document["score"]
        json.dump(document, open(f"{sys.argv[2]}/{file_name}.json", "w"))
