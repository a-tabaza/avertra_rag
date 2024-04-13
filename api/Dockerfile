## Python 3.11
FROM python:3.11 

## Set working directory
WORKDIR /app

## Install dependencies
COPY requirements.txt .

## Takes about an hour to run (thank you CUDA!)
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

## Copy scripts
COPY scripts/ scripts/

## Seed queries
COPY seed_keywords.txt .

## Fetch revelant documents (wikipedia)
## Usage: python3 fetch_documents.py <seed_queries_file> <output_folder>
RUN mkdir -p data/documents 
RUN python3 scripts/fetch_documents.py seed_keywords.txt data/documents

## Chunk documents
## Usage: python3 chunk.py <document_path> <output_folder> <output_file>
RUN mkdir -p data/chunked 
RUN python3 scripts/chunk.py data/documents data/chunked "chunks"

## Donwload model weights
## Usage: python3 download_weights.py <model_name>
RUN mkdir -p models
RUN python3 scripts/download_weights.py mixedbread-ai/mxbai-embed-large-v1

## Embed documents
## Usage: python3 embed.py <chunks_path> <output_folder> <output_file>
RUN mkdir -p data/embeddings
RUN python3 scripts/embed.py data/chunked/chunks.json data/embeddings "embeddings"

## Populate index
## Usage: python3 populate_index.py <embeddings_file> <output_folder> <output_file>
RUN mkdir -p data/indexes
RUN python3 scripts/populate_index.py data/embeddings/embeddings.npy data/indexes "index"

## Start server 
COPY api.py .
COPY utils.py .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]