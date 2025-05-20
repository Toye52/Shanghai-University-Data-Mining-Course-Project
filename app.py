import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
import json
import os
import numpy as np
from pymilvus import connections, Collection

# 模型路径
embedding_model_path = "./models--shibing624--text2vec-base-multilingual/snapshots/6633dc49e554de7105458f8f2e96445c6598e9d1"
translation_model_path = "Qwen/Qwen2.5-0.5B"
cross_encoder_path = "Alibaba-NLP/gte-large-zh"

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 Qwen 模型（用于中文回答）
qwen_tokenizer = AutoTokenizer.from_pretrained(translation_model_path, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(translation_model_path, trust_remote_code=True).to(device).eval()

# 加载 Bi-Encoder（中文向量模型）
embedding_model = SentenceTransformer(embedding_model_path)

# [Cross-Encoder] 加载 Cross-Encoder 模型
cross_encoder_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_path)
cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_path).to(device).eval()

# 连接 Milvus
connections.connect("default", host="localhost", port="19530")
collection = Collection("text_embeddings")

# 计算余弦相似度
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# 从 Milvus 查询相关文档
def retrieve_relevant_documents(query, embedding_model, top_k=5):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["content_chunk"]
    )
    return [(hit.score, hit.entity.get("content_chunk")) for hit in results[0]]

# [Cross-Encoder] 对候选文本块进行重排序
def rerank_with_cross_encoder(query, candidates, top_k=10):
    pairs = [(query, passage) for _, passage in candidates]
    try:
        inputs = cross_encoder_tokenizer.batch_encode_plus(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(device)
        with torch.no_grad():
            scores = cross_encoder_model(**inputs).logits.squeeze(-1)
        scores = scores.cpu().numpy()
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        reranked_docs = [candidates[i] for i in sorted_indices]
        return reranked_docs
    except Exception as e:
        print(f"[⚠️ 精排失败] 错误: {e}")
        return candidates[:top_k]

# 生成回答（直接用中文回答）
def generate_answer_with_llm(query, relevant_docs, history=None):
    context = " ".join([doc[1] for doc in relevant_docs])
    history_context = ""
    # 新增：判断与历史的相关性，仅在强相关时才纳入历史
    use_history = False
    if history and len(history) > 0:
        # 计算当前query与历史每条user query的相似度
        query_emb = embedding_model.encode(query, normalize_embeddings=True)
        similarities = []
        for item in history:
            prev_query = item[0]
            prev_emb = embedding_model.encode(prev_query, normalize_embeddings=True)
            sim = cosine_similarity(query_emb, prev_emb)
            similarities.append(sim)
        # 设定阈值，只有有一条历史与当前query相似度大于0.7才纳入历史
        if max(similarities) > 0.7:
            use_history = True

    if use_history:
        history_context = "\n".join([f"用户: {item[0]}\n助手: {item[1]}" for item in history])
    else:
        history_context = ""

    prompt = f"""请仅基于以下内容回答用户的问题，如果找不到答案，请明确说明。不要编造信息。

上下文内容：
{context}

对话历史：
{history_context}

用户提问：{query}

回答："""
    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(device)
    try:
        with torch.no_grad():
            outputs = qwen_model.generate(**inputs, max_new_tokens=512)
        answer = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("回答：")[-1].strip()
    except Exception as e:
        print(f"[⚠️ 生成回答失败] 错误：{e}")
        answer = "抱歉，无法生成回答。"

    return answer

# 主处理流程
def process_query(query, embedding_model, history=None):
    candidates = retrieve_relevant_documents(query, embedding_model, top_k=100)
    reranked_docs = rerank_with_cross_encoder(query, candidates, top_k=10)
    answer = generate_answer_with_llm(query, reranked_docs)
    return answer, reranked_docs

# Streamlit 前端
st.title("中文向量检索问答系统")

with st.sidebar:
    st.header("模型信息")
    st.write("向量模型: text2vec-base-multilingual")
    st.write("回答模型: Qwen2.5-0.5B")
    st.write("精排模型: Alibaba-NLP/gte-large-zh")

query = st.text_input("请输入您的问题:")

if "history" not in st.session_state:
    st.session_state.history = []  # 初始化对话历史

# 显示对话历史
history_container = st.container()
with history_container:
    for idx, (user_query, bot_response) in enumerate(st.session_state.history):
        st.markdown(f"**用户 {idx + 1}:** {user_query}")
        st.markdown(f"**助手 {idx + 1}:** {bot_response}")
        st.markdown("---")

# 用户输入
input_container = st.container()
with input_container:
    query = st.text_input("请输入您的问题:", key="user_input")
    if st.button("提交"):
        if query:
            # 处理查询，并获取回答和相关文档
            answer, relevant_docs = process_query(query, embedding_model, history=st.session_state.history)

            # 保存历史记录
            st.session_state.history.append((query, answer))

            # 显示相关文档
            st.subheader("相关文档：")
            for idx, doc in enumerate(relevant_docs):
                st.write(f"**文档 {idx + 1}:**")
                st.write(f"内容: {doc[1]}")
                st.write("---")

            # 显示回答
            st.subheader("回答：")
            st.write(answer)
        else:
            st.warning("请输入问题后再提交")