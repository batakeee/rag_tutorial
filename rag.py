import torch
import os
import faiss
import pickle
import pandas as pd
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config.prompt import *
from config.path import *
from config.key import API_KEY
import warnings

warnings.simplefilter('ignore')


# *******************************
# 設定
# *******************************
os.environ['OPENAI_API_KEY'] = API_KEY
llm = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0)
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

def get_embeddings(texts) -> torch.Tensor:
    """
    文章を1024次元のベクトルに変換し、L2正規化を適用
    """
    embedding = embedding_model.encode(texts)
    embedding = F.normalize(torch.tensor(embedding), p=2, dim=1)

    return embedding.numpy()

def retrieve(query, k=4)-> list[str]:
    """
    - query: 検索クエリ
    - k: 取得する文の数
    """
    # クエリの埋め込みを取得
    query_embedding = get_embeddings([query])
    _, indices = faiss_index.search(query_embedding.reshape(1, -1), k)
    results = []
    for idx in indices[0]:
        chunk = chunk_store[idx]
        results.append(chunk)
    return results

# *******************************
# 外部知識のロード
# *******************************
faiss_index = faiss.read_index(FAISS_FILE)
with open(PICKLE_FILE, "rb") as f:
    chunk_store = pickle.load(f)


# *******************************
# Chainを作成
# *******************************
augmented_prompt = PromptTemplate(
    template=AUGMENT_TEMPLATE,
    input_variables=["context1", "context2", "context3", "context4", "question"]
)
chain = LLMChain(llm=llm, prompt=augmented_prompt)


# *******************************
# 回答生成
# *******************************

df = pd.read_csv(INPUT_FILE)

for index, row in df.iterrows():
    # 検索
    docs = retrieve(row["Question"])

    input_data = {
        "context1": docs[0],
        "context2": docs[1],
        "context3": docs[2],
        "context4": docs[3],
        "question": row["Question"]
    }
    # 回答生成
    ans = chain.run(input_data)

    # 書き込み
    df.at[index, "Response"] = ans
    df.at[index, "Prompt"] = augmented_prompt.format(**input_data)

# 出力
df.to_csv(OUTPUT_FILE, index=False)