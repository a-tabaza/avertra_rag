"""
This script downloads the weights for the specified model from the Hugging Face Hub and saves them in the models folder.
"""

from huggingface_hub import snapshot_download
import sys

if len(sys.argv) < 2:
    print(
        "Usage: python download_weights.py <model_name> \nAvailable models: mixedbread-ai/mxbai-embed-large-v1, mixedbread-ai/mxbai-rerank-xsmall-v1"
    )
    exit(1)

if __name__ == "__main__":
    model_name = sys.argv[1]
    if model_name not in [
        "mixedbread-ai/mxbai-embed-large-v1",
        "mixedbread-ai/mxbai-rerank-xsmall-v1",
    ]:
        print(
            "Usage: python download_weights.py <model_name> \nAvailable models: mixedbread-ai/mxbai-embed-large-v1, mixedbread-ai/mxbai-rerank-xsmall-v1"
        )
        exit(1)
    snapshot_download(
        repo_id=model_name,
        local_dir=f"models/{model_name.split('/')[-1]}",
        local_dir_use_symlinks=False,
        ignore_patterns=[
            "mxbai-embed-large-v1-f16.gguf",
            "model.onnx",
            "model_fp16.onnx",
            "model_quantized.onnx",
        ],
    )
    print(f"Downloaded weights for model {model_name} at {model_name.split('/')[-1]}")
