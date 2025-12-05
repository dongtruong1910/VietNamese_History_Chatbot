# 🇻🇳 Sử Việt AI - Trợ Lý Lịch Sử Việt Nam Thông Minh

**Sử Việt AI** là một Chatbot chuyên sâu về lịch sử Việt Nam, kết hợp sức mạnh của **Generative AI** (mô hình ngôn ngữ lớn) và kỹ thuật **Hybrid RAG** (Retrieval-Augmented Generation) để đưa ra câu trả lời chính xác, trung thực và có dẫn chứng.

Dự án được xây dựng dựa trên mô hình **Qwen2.5-1.5B**, được tinh chỉnh (Fine-tune) qua 2 giai đoạn **SFT** (Supervised Fine-Tuning) và **DPO** (Direct Preference Optimization).

---

## 🌟 Tính Năng Nổi Bật

* **Hybrid Search RAG:** Kết hợp tìm kiếm theo ngữ nghĩa (Vector Search với FAISS) và tìm kiếm từ khóa (Keyword Search với BM25) để tối ưu hóa khả năng truy xuất thông tin.
* **Reranking:** Sử dụng Cross-Encoder (BGE-M3) để chấm điểm và sắp xếp lại các văn bản tìm được, đảm bảo ngữ cảnh tốt nhất cho AI.
* **Mô hình tối ưu:** Sử dụng Qwen2.5-1.5B đã được train DPO, giúp câu trả lời tự nhiên, mượt mà và tuân thủ chỉ thị tốt hơn.
* **Strict Mode:** Cơ chế "Kỷ luật sắt" giúp giảm thiểu ảo giác (hallucination), yêu cầu AI chỉ trả lời dựa trên thông tin tìm thấy.
* **Giao diện thân thiện:** Tích hợp Gradio với giao diện Dark Mode phong cách lịch sử.

---

## 📂 Cấu Trúc Dự Án

```text
VietNamese_History_Chatbot/
├── .venv/                   # Môi trường ảo (Virtual Env)
├── data/                    # Chứa dữ liệu thô (json/jsonl)
│   └── rawhistory_200k.jsonl
├── models/                  # Chứa các Adapter sau khi train
│   └── DPO_Final_Model/     # <-- Copy file model DPO vào đây
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── notebooks/               # Mã nguồn huấn luyện (Colab)
│   ├── SFT_Trainer.ipynb    # Code train giai đoạn 1
│   └── DPO_Trainer.ipynb    # Code train giai đoạn 2
├── rag_builder/             # Module xây dựng Vector Database
│   ├── build_db.py
│   └── results/             # Nơi chứa file index FAISS & metadata
├── app.py                   # Ứng dụng chính (Gradio Web UI)
├── requirements.txt         # Danh sách thư viện cần thiết
└── README.md                # Tài liệu hướng dẫn
```
## 📚 Dữ liệu (Datasets)

Dự án sử dụng các bộ dữ liệu chất lượng cao về **lịch sử Việt Nam**:

### 🔹 Dữ liệu Fine-tune (SFT)
**Dataset:** `minhxthanh/Vietnam-History-15k`  
**Mục đích:**  
- Huấn luyện mô hình cách trả lời.  
- Học văn phong tự nhiên.  
- Trang bị kiến thức nền tảng về lịch sử Việt Nam.

### 🔹 Dữ liệu RAG (Knowledge Base)
**Dataset:** `minhxthanh/Vietnam-History-200K-Vi`  
**Mục đích:**  
- Tạo **Vector Database**.  
- Làm kho tri thức để mô hình truy vấn bằng RAG, đảm bảo trả lời chính xác và có nguồn.
