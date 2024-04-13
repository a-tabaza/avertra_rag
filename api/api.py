from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import json
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from usearch.index import Index

from utils import transform_query

index = Index.restore("/app/data/indexes/index.usearch")
chunks = json.load(open("/app/data/chunked/chunks.json", "r"))
model = SentenceTransformer("/app/models/mxbai-embed-large-v1")
ranker = Ranker()

app = FastAPI(docs_url="/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RagQuery(BaseModel):
    query: str


@app.post("/embed/")
async def query(query: RagQuery):
    try:
        embedding = model.encode([transform_query(query.query)]).squeeze(0)
        if embedding is None:
            raise ValueError("embedding is None")
        return {"embedding": embedding.tolist()}
    except Exception as e:
        error_message = "Internal Server Error: {}: {}".format(type(e).__name__, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message
        ) from e


@app.post("/retrieve/")
async def query(query: RagQuery, k: int = 5):
    
    if k <= 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="k must be at least 3",
        )
    if k > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="k must be at most 10",
        )
    try:
        query_embedding = model.encode([transform_query(query.query)]).squeeze(0)
        if query_embedding is None:
            raise ValueError("query embedding is None")

        results = index.search(query_embedding, count=30)
        if results is None:
            raise ValueError("search results are None")

        ids = [r[0] for r in results.to_list()]
        if not ids:
            raise ValueError("no search results")

        reranker_batch = [
            {
                "id": i,
                "text": chunks[i]["chunk_text"],
                "meta": {},
            }
            for i in ids
        ]
        rerank_req = RerankRequest(query=query.query, passages=reranker_batch)
        res = ranker.rerank(rerank_req)
        results_ = [result["text"] for result in res[:k]]
        if results_ is None:
            raise ValueError("search results are None")
        return {"results": results_}
    except Exception as e:
        error_message = "Internal Server Error: {}: {}".format(type(e).__name__, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_message
        ) from e
