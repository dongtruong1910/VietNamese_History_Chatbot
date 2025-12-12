import os
import torch
import gradio as gr
import faiss
import pickle
import numpy as np
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sentence_transformers import SentenceTransformer, CrossEncoder
from threading import Thread
from rank_bm25 import BM25Okapi
import wikipedia

# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_ID = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

# Đường dẫn Database
RAG_INDEX_PATH = os.path.join(BASE_DIR, "rag_builder", "results", "history_vector.index")
RAG_META_PATH = os.path.join(BASE_DIR, "rag_builder", "results", "history_metadata.pkl")

# Cấu hình Logic
RERANK_TOP_K = 5  # Chỉ lấy 5 kết quả tốt nhất để rerank
RAG_THRESHOLD = 0.9

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Main DeGvice: {device.upper()}")

# ==============================================================================
# 2. LOAD COMPONENT
# ==============================================================================
try:
    print("⏳ Loading Models (Embedder & Reranker) on CPU to save VRAM...")
    embedder = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device=device)
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device=device)
    print("✅ Embedder & Reranker Ready (CPU Mode)!")
except Exception as e:
    print(f"❌ Lỗi load Embedder/Reranker: {e}")
    embedder, reranker = None, None

rag_index = None
rag_metadata = []
bm25 = None

if os.path.exists(RAG_INDEX_PATH):
    print("⏳ Loading Database...")
    rag_index = faiss.read_index(RAG_INDEX_PATH)
    with open(RAG_META_PATH, "rb") as f:
        rag_metadata = pickle.load(f)

    print("⏳ Initializing BM25 Search...")
    try:
        tokenized_corpus = [doc['original_query'].lower().split() for doc in rag_metadata]
        bm25 = BM25Okapi(tokenized_corpus)
        print("✅ Hybrid Search Ready (BM25 Loaded)!")
    except Exception as e:
        print(f"❌ Lỗi BM25: {e}")

# ==============================================================================
# LOAD QWEN-3B-4BIT
# ==============================================================================
print(f"⏳ Loading Qwen-3B-4bit from {MODEL_ID}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print(f"✅ Model 3B Ready! (VRAM optimized)")

    # Dọn dẹp RAM sau khi load
    torch.cuda.empty_cache()
    gc.collect()

except Exception as e:
    print(f"❌ Model Error: {e}")
    model = None


# ==============================================================================
# 3. HÀM PHỤ TRỢ: TÌM KIẾM WEB & HYBRID
# ==============================================================================
# ==============================================================================
# 3. CHỨC NĂNG THÔNG MINH: SUY LUẬN TỪ KHÓA & TRA WIKI
# ==============================================================================
def extract_keywords(user_query):
    """
    dùng Python cắt bỏ các từ để hỏi thừa thãi.
    """
    print(f"⚡ Đang lọc từ khóa nhanh cho: {user_query}")

    # Danh sách các từ rác thường gặp trong câu hỏi
    stop_phrases = [
        "cho tôi hỏi", "cho mình hỏi", "bạn có biết", "hãy cho biết",
        "là gì", "là ai", "như thế nào", "tại sao", "khi nào", "ở đâu", "bao nhiêu",
        "ý nghĩa của", "nguyên nhân", "diễn biến", "kết quả", "tóm tắt",
        "có", "những", "các", "cái", "gì", "?", "!",
        # THÊM CÁC TỪ NỐI NÀY:
        "trong", "cuộc", "của", "về", "việc", "đã", "đang", "sẽ", "ở", "tại", "bị", "được"
    ]

    # 1. Chuyển về chữ thường để xử lý
    clean_text = user_query.lower()

    # 2. Xóa các từ rác
    for phrase in stop_phrases:
        clean_text = clean_text.replace(phrase, "")

    # 3. Chuẩn hóa lại (xóa khoảng trắng thừa)
    clean_text = " ".join(clean_text.split())

    # 4. Nếu xóa hết trơn (câu hỏi quá ngắn), thì lấy lại câu gốc
    if len(clean_text) < 2:
        clean_text = user_query

    print(f"👉 Từ khóa nhanh: '{clean_text}'")
    return clean_text


def search_wikipedia(query):
    """Tìm Top 1 bài và Đọc 3000 ký tự đầu tiên từ Wikipedia Tiếng Việt."""
    print(f"🌐 Tra cứu Wiki (Deep Read): {query}")
    wikipedia.set_lang("vi")

    try:
        # 1. Tìm tiêu đề bài viết khớp nhất
        search_results = wikipedia.search(query, results=1)

        if not search_results:
            print("   --> Wiki không tìm thấy bài nào.")
            return ""

        title = search_results[0]
        print(f"   --> Đang đọc bài: {title}")

        # 2. Truy cập vào trang để lấy nội dung đầy đủ
        page = wikipedia.page(title, auto_suggest=False)

        # 3. Lấy 3000 ký tự đầu tiên (Chứa Intro + Infobox + Chương 1)
        content = page.content[:3000]

        # Xử lý xuống dòng cho gọn
        clean_content = content.replace("\n\n", "\n")

        return f"NGUỒN: Wikipedia Tiếng Việt ({title})\nNỘI DUNG TRÍCH DẪN:\n{clean_content}..."

    except wikipedia.DisambiguationError as e:
        # Nếu từ khóa chung chung, lấy bài đầu tiên trong gợi ý
        try:
            first_opt = e.options[0]
            page = wikipedia.page(first_opt, auto_suggest=False)
            content = page.content[:2000]
            return f"NGUỒN: Wikipedia ({first_opt})\nNỘI DUNG:\n{content}..."
        except:
            return ""

    except Exception as e:
        print(f"❌ Lỗi Wiki: {e}")
        return ""
def hybrid_search(query, top_k=15):
    vec_candidates = []
    if rag_index and embedder:
        vec = embedder.encode([query])
        D, I = rag_index.search(np.array(vec).astype('float32'), k=top_k)
        for idx in I[0]:
            if idx != -1:
                vec_candidates.append(idx)

    bm25_candidates = []
    if bm25:
        try:
            tokenized_query = query.lower().split()
            doc_scores = bm25.get_scores(tokenized_query)
            top_n = np.argsort(doc_scores)[::-1][:top_k]
            bm25_candidates = top_n.tolist()
        except:
            pass

    all_indexes = list(set(vec_candidates + bm25_candidates))

    final_candidates = []
    seen_answers = set()

    for idx in all_indexes:
        if idx < len(rag_metadata):
            item = rag_metadata[idx]
            ans = item.get('answer', '').strip()
            if ans not in seen_answers and len(ans) > 10:
                seen_answers.add(ans)
                final_candidates.append([query, ans])

    return final_candidates


# ==============================================================================
# 4. BOT RESPONSE (LOGIC THÔNG MINH)
# ==============================================================================
def bot_response(message, history):
    if model is None:
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ Lỗi: Model chưa được load"})
        return history, ""

    final_context = ""
    source_label = ""

    # Dọn dẹp VRAM trước khi sinh
    torch.cuda.empty_cache()

    try:
        # BƯỚC 1: Tìm trong RAG nội bộ
        candidates = hybrid_search(message, top_k=15)

        if candidates and reranker:
            scores = reranker.predict(candidates)
            scored_candidates = []
            for i, score in enumerate(scores):
                scored_candidates.append({"score": score, "text": candidates[i][1]})

            scored_candidates.sort(key=lambda x: x['score'], reverse=True)

            # Lấy Top 5
            top_candidates = scored_candidates[:RERANK_TOP_K]

            if top_candidates:
                best = top_candidates[0]
                # BƯỚC 2: Kiểm tra ngưỡng điểm
                if best['score'] > RAG_THRESHOLD:
                    final_context = best['text']
                    source_label = "📚 Dữ liệu nội bộ"
                    print(f"✅ CHỐT RAG (Score {best['score']:.2f} > {RAG_THRESHOLD})")
                else:
                    print(f"⚠️ Điểm RAG thấp ({best['score']:.2f} < {RAG_THRESHOLD}) -> Chuyển sang Web Search...")


                    wiki_keyword = extract_keywords(message)


                    wiki_content = search_wikipedia(wiki_keyword)

                    if wiki_content:
                        final_context = wiki_content
                        source_label = f"🌐 Wikipedia (Từ khóa: {wiki_keyword})"
                        print("✅ WIKI SUCCESS")
                    else:
                        print("❌ Wiki failed.")
    except Exception as e:
        print(f"Quy trình Search gặp lỗi: {e}")

    # Tạo Prompt
    if final_context:
        prompt = f"""### TÀI LIỆU THAM KHẢO:
    {final_context}

    ### CHỈ THỊ TUYỆT ĐỐI:
    Bạn là trợ lý lịch sử người Việt. Hãy trả lời câu hỏi theo quy tắc:
    1. **Nguồn tin:** Ưu tiên dùng [TÀI LIỆU] (Nguồn: {source_label}).
    2. **Bổ sung:** Nếu tài liệu thiếu, hãy dùng kiến thức của bạn và nói "Theo kiến thức của tôi...".
    3. **Ngôn ngữ:** CHỈ DÙNG TIẾNG VIỆT. Cấm tuyệt đối tiếng Trung/Anh.
    4. **Văn phong:** Ngắn gọn, súc tích, đi thẳng vào câu trả lời.

    ### CÂU HỎI:
    {message}

    ### TRẢ LỜI:"""
    else:
        # Fallback (Khi không tìm thấy gì cả)
        prompt = f"""Bạn là trợ lý lịch sử người Việt.
    Nhiệm vụ: Trả lời câu hỏi ngắn gọn, chính xác bằng Tiếng Việt.
    Lưu ý: Nếu không biết, hãy nói "Tôi không biết". KHÔNG ĐƯỢC BỊA ĐẶT hay dùng tiếng nước ngoài.

    Câu hỏi: {message}
    Trả lời:"""

    # Qwen Chat Template
    messages = [
        {"role": "system", "content": "Bạn là trợ lý AI hữu ích."},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=False,  # Greedy Search để trung thực
        pad_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    if history is None:
        history = []

    history.append({"role": "user", "content": message})

    partial_text = f"({source_label})\n" if final_context else ""
    for new_text in streamer:
        partial_text += new_text
        temp_history = history[:-1] + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": partial_text}
        ]
        yield temp_history, ""

    history.append({"role": "assistant", "content": partial_text})
    yield history, ""


# ==============================================================================
# 5. CUSTOM CSS (GIỮ NGUYÊN UI CỦA BẠN)
# ==============================================================================
custom_css = """
<style>
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 2rem auto !important;
}

.contain {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
    padding: 2rem !important;
}

h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    text-align: center !important;
    margin-bottom: 0.5rem !important;
}

.chatbot {
    border-radius: 15px !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
}

button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

textarea {
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    padding: 12px !important;
    font-size: 1rem !important;
}

textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    outline: none !important;
}

.footer {
    text-align: center;
    color: #666;
    margin-top: 2rem;
    font-size: 0.9rem;
}

.example-btn {
    background: white !important;
    color: #667eea !important;
    border: 2px solid #667eea !important;
    margin: 5px !important;
}

.example-btn:hover {
    background: #667eea !important;
    color: white !important;
}
</style>
"""

# ==============================================================================
# 6. GRADIO INTERFACE
# ==============================================================================
with gr.Blocks() as demo:
    gr.HTML(custom_css)

    gr.Markdown(
        """
        # 🇻🇳 Sử Việt AI - Trợ Lý Lịch Sử Thông Minh
        ### Hệ thống Hybrid RAG + Qwen 3B (Web Fallback)
        *Tìm kiếm kết hợp Vector Search & BM25 | Reranking với Cross-Encoder*
        """
    )

    chatbot = gr.Chatbot(label="💬 Trò chuyện", height=500)

    with gr.Row():
        msg = gr.Textbox(
            label="",
            placeholder="💭 Hỏi tôi về lịch sử Việt Nam...",
            show_label=False,
            lines=2
        )

    with gr.Row():
        submit = gr.Button("📤 Gửi", variant="primary")
        clear = gr.Button("🗑️ Xóa")

    gr.Markdown("### 💡 Câu hỏi gợi ý:")
    with gr.Row():
        example1 = gr.Button("Bác Hồ ra đi tìm đường cứu nước năm nào?", elem_classes="example-btn")
        example2 = gr.Button("Ý nghĩa chiến thắng Điện Biên Phủ trên không?", elem_classes="example-btn")

    gr.Markdown(
        """
        <div class='footer'>
        💡 <b>Mẹo sử dụng:</b> Đặt câu hỏi cụ thể về lịch sử Việt Nam để nhận câu trả lời chính xác nhất<br>
        ⚡ Powered by Qwen 2.5-3B + Vietnamese Embedder + Reranker
        </div>
        """
    )

    # Event handlers
    msg.submit(bot_response, [msg, chatbot], [chatbot, msg])
    submit.click(bot_response, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

    example1.click(lambda: "Bác Hồ ra đi tìm đường cứu nước năm nào?", None, msg)
    example2.click(lambda: "Ý nghĩa chiến thắng Điện Biên Phủ trên không?", None, msg)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )