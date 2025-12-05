import json
import faiss
import numpy as np
import pickle
import os
import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# CẤU HÌNH
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
# Sửa tên file này cho đúng với file bạn đang có (json hay jsonl)
DATA_PATH = os.path.join(current_dir, '..', 'data', 'rawhistory_200k.jsonl')

OUTPUT_INDEX_PATH = os.path.join(current_dir, 'results', 'history_vector.index')
OUTPUT_META_PATH = os.path.join(current_dir, 'results', 'history_metadata.pkl')


# ---------------------------------------------------------
# 1. HÀM ĐỌC FILE (CHẤP NHẬN MỌI ĐỊNH DẠNG)
# ---------------------------------------------------------
def load_and_process(file_path):
    print(f"📂 Đang đọc file: {file_path}")

    documents = []
    metadata = []

    total_lines = 0
    valid_pairs = 0

    # Đọc theo kiểu JSON Lines (Mỗi dòng 1 object) - Phổ biến với file lớn
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            total_lines += 1
            try:
                # Nếu file là mảng JSON lớn [] thì dòng này sẽ lỗi, ta xử lý ở except
                item = json.loads(line)

                # --- LOGIC TRÍCH XUẤT (NỚI LỎNG) ---
                msgs = item.get("messages", [])
                user_q = ""
                assist_a = ""

                for m in msgs:
                    if m['role'] == 'user':
                        user_q = m['content']
                    elif m['role'] == 'assistant':
                        # LẤY LUÔN! Không cần check 'final' nữa
                        # Nếu có nhiều câu trả lời, lấy câu cuối cùng (thường là câu chốt)
                        assist_a = m['content']

                if user_q and assist_a:
                    documents.append(user_q)
                    metadata.append({
                        "original_query": user_q,
                        "answer": assist_a
                    })
                    valid_pairs += 1

            except json.JSONDecodeError:
                # Trường hợp file là một cục JSON Array lớn [...]
                pass

    # Nếu đọc từng dòng thất bại (valid_pairs = 0), thử đọc kiểu JSON Array toàn cục
    if valid_pairs == 0:
        print("⚠️ Đọc theo dòng không được, chuyển sang đọc toàn bộ file (JSON Array)...")
        f = open(file_path, 'r', encoding='utf-8')
        try:
            items = json.load(f)
            total_lines = len(items)
            for item in items:
                msgs = item.get("messages", [])
                user_q = ""
                assist_a = ""
                for m in msgs:
                    if m['role'] == 'user':
                        user_q = m['content']
                    elif m['role'] == 'assistant':
                        assist_a = m['content']

                if user_q and assist_a:
                    documents.append(user_q)
                    metadata.append({"original_query": user_q, "answer": assist_a})
                    valid_pairs += 1
        except Exception as e:
            print(f"❌ Lỗi đọc file: {e}")
        finally:
            f.close()

    print(f"{'=' * 30}")
    print(f"📊 TỔNG KẾT DỮ LIỆU:")
    print(f"   - Tổng số dòng/item đã quét: {total_lines}")
    print(f"   - Số cặp Q-A hợp lệ lấy được: {valid_pairs}")
    print(f"{'=' * 30}")

    return documents, metadata


# ---------------------------------------------------------
# 2. CHẠY BUILD
# ---------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")

    # A. Xử lý dữ liệu
    docs, metas = load_and_process(DATA_PATH)

    if not docs:
        print("❌ Không lấy được dữ liệu nào!")
        return

    # B. Load Model
    print("🚀 Loading Embedder...")
    embedder = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)

    # C. Encode
    print(f"⚡ Đang mã hóa {len(docs)} câu (Batch 128)...")
    embeddings = embedder.encode(docs, batch_size=128, show_progress_bar=True, convert_to_numpy=True)

    # D. Save FAISS
    print("📦 Creating Index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, OUTPUT_INDEX_PATH)
    with open(OUTPUT_META_PATH, "wb") as f:
        pickle.dump(metas, f)

    print("🎉 XONG!")


if __name__ == "__main__":
    main()