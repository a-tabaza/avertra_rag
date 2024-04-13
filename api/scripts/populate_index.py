import sys
from tqdm import tqdm
import numpy as np
from usearch.index import Index

EMBEDDING_DIM = 1024

index = Index(
    ndim=EMBEDDING_DIM,
    metric="cos",
    dtype="f32",
    connectivity=16,
    expansion_add=128,
    expansion_search=64,
    multi=False,
)

if len(sys.argv) < 3:
    print(
        "Usage: python populate_index.py <embeddings_file> <output_folder> <output_file>"
    )
    exit(1)

if __name__ == "__main__":
    embeddings = np.load(sys.argv[1])
    for i, embedding in enumerate(tqdm(embeddings)):
        index.add(i, embedding)
    index.save(f"{sys.argv[2]}/{sys.argv[3]}.usearch")
