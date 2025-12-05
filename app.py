import os
import torch
import gradio as gr
import faiss
import pickle
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from threading import Thread
from rank_bm25 import BM25Okapi

# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(BASE_DIR, "models", "DPO_Final_Model")
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

RAG_INDEX_PATH = os.path.join(BASE_DIR, "rag_builder", "results", "history_vector.index")
RAG_META_PATH = os.path.join(BASE_DIR, "rag_builder", "results", "history_metadata.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Device: {device.upper()}")

# ==============================================================================
# 2. LOAD COMPONENT
# ==============================================================================
try:
    print("⏳ Loading Models (Embedder & Reranker)...")
    embedder = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device=device)
except:
    print("❌ Lỗi load Embedder/Reranker")
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

print(f"⏳ Loading GenAI Model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=torch.float16, device_map=device)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    print("✅ Model Ready!")
except Exception as e:
    print(f"❌ Model Error: {e}")
    model = None


# ==============================================================================
# 3. HÀM PHỤ TRỢ: HYBRID SEARCH
# ==============================================================================
def hybrid_search(query, top_k=15):
    vec_candidates = []
    if rag_index and embedder:
        vec = embedder.encode([query])
        D, I = rag_index.search(np.array(vec).astype('float32'), k=top_k)
        for idx in I[0]:
            if idx != -1: vec_candidates.append(idx)

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
            if ans not in seen_answers and len(ans) > 5:
                seen_answers.add(ans)
                final_candidates.append([query, ans])

    return final_candidates


# ==============================================================================
# 4. BOT RESPONSE - MODIFIED FOR GRADIO CHATBOT
# ==============================================================================
def bot_response(message, history):
    if model is None:
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "❌ Lỗi: Model chưa được load"})
        return history, ""

    rag_context = ""

    try:
        candidates = hybrid_search(message, top_k=20)

        if candidates and reranker:
            scores = reranker.predict(candidates)
            scored_candidates = []
            for i, score in enumerate(scores):
                scored_candidates.append({"score": score, "text": candidates[i][1]})

            scored_candidates.sort(key=lambda x: x['score'], reverse=True)

            if scored_candidates:
                best = scored_candidates[0]
                if best['score'] > -1.5:
                    rag_context = best['text']
                    print(f"✅ CHỐT RAG (Score {best['score']:.2f}): {rag_context[:50]}...")
                else:
                    print("⚠️ Điểm thấp, không lấy context.")
    except Exception as e:
        print(f"Search Error: {e}")

    if rag_context:
        prompt = f"""### CHỈ THỊ:
1. Bạn là trợ lý AI. Nhiệm vụ là trích xuất thông tin từ "BỐI CẢNH" để trả lời.
2. TUYỆT ĐỐI KHÔNG sử dụng kiến thức bên ngoài.
3. Nếu BỐI CẢNH nói là A, hãy trả lời là A.

### BỐI CẢNH:
"{rag_context}"

### CÂU HỎI:
{message}

### TRẢ LỜI:"""
    else:
        prompt = message

    messages = [
        {"role": "system", "content": "Bạn là trợ lý AI trung thực. Chỉ trả lời dựa trên dữ liệu được cung cấp."},
        {"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=256,
        temperature=0.1,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Ensure history is a list
    if history is None:
        history = []

    # Add user message
    history.append({"role": "user", "content": message})

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        # Create temporary history with assistant response
        temp_history = history[:-1] + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": partial_text}
        ]
        yield temp_history, ""

    # Final update
    history.append({"role": "assistant", "content": partial_text})
    yield history, ""


# ==============================================================================
# 5. CUSTOM CSS
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
        ### Hệ thống Hybrid RAG + DPO Fine-tuned Model
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
        ⚡ Powered by Qwen 2.5-1.5B + Vietnamese Embedder + Reranker
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
        server_name="127.0.0.1",  # Hoặc dùng "0.0.0.0" nếu muốn truy cập từ máy khác trong mạng LAN
        server_port=7860,
        share=False  # Đổi thành True nếu muốn link public trên internet
    )