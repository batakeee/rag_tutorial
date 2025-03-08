# rag_tutorial
Source Code for Playing with Naive RAG (Retrieval-Augmented Generation)

# Requirements
Since `sentence_transformers` or its dependencies (`torch` or `transformers`) are not compatible with `Numpy 2.x`, we will install `Numpy 1.x` instead.
```bash
pip install sentence_transformers langchain-community faiss-cpu pandas openai numpy==1.26.4
```
# Configuration
1. Set your `API_KEY` in `config/key.py`.
2. If necessary, adjust the paths in `config/path.py` accordingly.

# Usage
Run the following command to execute the script:
```bash
python rag.py
```

