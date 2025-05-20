from pathlib import Path
from bs4 import BeautifulSoup
import json
import re
from typing import List

# 配置参数
HTML_DIR = Path("./milvus_data/html")  # HTML文件目录
OUTPUT_JSON = "./processed_data.json"  # 输出文件路径
CHUNK_SIZE = 500  # 文本块大小（字符数）
CHUNK_OVERLAP = 50  # 块间重叠字符数

# 正则表达式预编译
CHINESE_CHAR_REGEX = re.compile(r'[\u4e00-\u9fa5]+')  # 匹配中文字符
URL_REGEX = re.compile(r'https?://\S+')  # 匹配URL链接


def split_text(text: str) -> List[str]:
    """
    按句子切分并合并为接近 CHUNK_SIZE 的块，重叠部分为前一块最后的1~2句（不超过50字符）。
    """
    # 分句（支持中英文）
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_len = len(sentence)

        if current_len + sentence_len <= CHUNK_SIZE:
            current_chunk.append(sentence)
            current_len += sentence_len
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # 新块开始
            current_chunk = [sentence]
            current_len = sentence_len

    # 添加最后一块
    if current_chunk:
        chunks.append(current_chunk)

    # 添加重叠（保留上一块最后1~2句，总长≤50字符）
    overlapped_chunks = []
    for i, chunk_sentences in enumerate(chunks):
        if i == 0:
            overlapped_chunks.append(''.join(chunk_sentences))
        else:
            # 获取上一块的结尾句子用于 overlap
            prev_sentences = chunks[i - 1]
            overlap = ""
            for s in reversed(prev_sentences[-2:]):  # 最多取2句
                if len(overlap) + len(s) <= CHUNK_OVERLAP:
                    overlap = s + overlap
                else:
                    break
            # 拼接 overlap + 当前块
            new_chunk = overlap + ''.join(chunk_sentences)
            overlapped_chunks.append(new_chunk)

    return overlapped_chunks



def process_html(html_path: Path) -> List[dict]:
    """
    处理单个HTML文件并返回分块结果
    """
    try:
        print(f"正在处理: {html_path.name}")
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        title = soup.select_one('h1.rich_media_title a').get_text(strip=True) if soup.select_one(
            'h1.rich_media_title a') else "无标题"
        publish_time = soup.find('em', id='publish_time').get_text(strip=True) if soup.find('em',
                                                                                            id='publish_time') else "未知时间"

        content_paragraphs = [
            p.get_text(strip=True)
            for p in soup.select('content p')
            if p.get_text(strip=True)
        ]
        full_text = '\n'.join(content_paragraphs)

        chunks = split_text(full_text)
        print(f" - 标题: {title}")
        print(f" - 时间: {publish_time}")
        print(f" - 分块数: {len(chunks)}")

        return [{
            "title": title,
            "publish_time": publish_time,
            "content_chunk": chunk,
            "chunk_id": f"{html_path.stem}_{i}",
            "file_path": str(html_path),
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i, chunk in enumerate(chunks)]

    except Exception as e:
        print(f"处理文件 {html_path.name} 失败: {str(e)}")
        return []


def batch_process():
    """批量处理所有HTML文件"""
    processed_data = []
    all_files = list(HTML_DIR.glob("*.html"))
    print(f"共发现 {len(all_files)} 个 HTML 文件。")

    for i, html_file in enumerate(all_files, start=1):
        print(f"\n[{i}/{len(all_files)}] 文件名: {html_file.name}")
        chunks = process_html(html_file)
        processed_data.extend(chunks)

    print(f"\n所有文件处理完成，共生成 {len(processed_data)} 个文本块。")
    print(f"正在写入 JSON 文件到: {OUTPUT_JSON}")

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f,
                  ensure_ascii=False,
                  indent=2,
                  sort_keys=True)

    print("写入完成。")


if __name__ == "__main__":
    batch_process()
