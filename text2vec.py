import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# ✅ Increase timeout for downloading large files
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # Set timeout to 300 seconds

embedding_model_path = snapshot_download(
    repo_id="shibing624/text2vec-base-multilingual",
    cache_dir="D:\作业\数据挖掘\\4\\",
    local_files_only=True  # 避免重复下载
)

# ✅ 创建缓存目录
os.makedirs("./cache/embeddings", exist_ok=True)

# ✅ 加载中文嵌入模型
print("加载 text2vec 中文嵌入模型...")
embedding_model = SentenceTransformer(embedding_model_path)

# ✅ 嵌入 + 缓存函数
def embed_with_cache(text: str):
    cache_file = f"cache/embeddings/{hash(text)}.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    embedding = embedding_model.encode(text, normalize_embeddings=True).tolist()
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(embedding, f)
    return embedding

# ✅ 加载原始数据
with open("./processed_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ 限制数据到前 10000 条
data = data[:10000]

# ✅ 向量生成
results = []
print("开始生成嵌入向量...")
for item in tqdm(data):
    content = item["content_chunk"]
    item["embedding"] = embed_with_cache(content)
    results.append(item)

# ✅ 保存为 JSONL
os.makedirs("data", exist_ok=True)
with open("data/embeddings.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ 所有数据处理完成，结果保存在 data/embeddings.jsonl")

# ✅ 连接到 Milvus 数据库
connections.connect("default", host="localhost", port="19530")

# ✅ 定义 Milvus 集合
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content_chunk", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(results[0]["embedding"]))
]
schema = CollectionSchema(fields, description="Text embeddings collection")
collection = Collection(name="text_embeddings", schema=schema)

# ✅ 插入数据到 Milvus
print("写入数据到 Milvus 数据库...")
data_to_insert = [
    [item["content_chunk"] for item in results],  # content_chunk
    [item["embedding"] for item in results]      # embedding
]
collection.insert(data_to_insert)

# ✅ 创建索引
index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
collection.create_index(field_name="embedding", index_params=index_params)

# ✅ 持久化集合
collection.load()
print("✅ 所有数据已写入 Milvus 数据库")
