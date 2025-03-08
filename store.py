import faiss
import pickle
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.path import *

# Embeddingモデルをロード
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,      # チャンクの文字数
    chunk_overlap = 50,    # チャンクオーバーラップの文字数
)

def get_embeddings(texts: list[str]) -> torch.Tensor:
    """
    文章を1024次元のベクトルに変換し、L2正規化を適用
    """
    embedding = embedding_model.encode(texts)
    embedding = F.normalize(torch.tensor(embedding), p=2, dim=1)

    return embedding.numpy()

# ファイルを開いて読み込む
with open(STORY_TEXT, "r", encoding="utf-8") as file:
    content = file.read()
# 改行をなくす
content = content.replace("\n","")

# 200文字でテキストを分割
texts = text_splitter.split_text(content)

# faissを利用してベクトルを保存
faiss_index = faiss.IndexFlatIP(1024)
faiss_index.add(get_embeddings(texts))

# 保存
faiss.write_index(faiss_index, FAISS_FILE)
with open(PICKLE_FILE, "wb") as f:
    pickle.dump(texts, f)