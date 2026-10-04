# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Minh Tuấn  
**Mã số học viên:** 2A202602420  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026  

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

Bảng đối chiếu chi tiết giữa lý thuyết bài giảng Production RAG Pipeline và các module code đã thực thi:

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích thực tế trong Lab |
|----------------|--------|-------------|-------------------------------------------|
| **Semantic Chunking** | M1 | `chunk_semantic()` | - Dùng SentenceTransformers (`all-MiniLM-L6-v2`) tính toán cosine similarity giữa các câu liên tiếp.<br>- Phân tách câu bằng `re.split(r'(?<=[.!?\n])\s+')`.<br>- Với threshold = 0.85, văn bản được ngắt tại đúng ranh giới chuyển đổi ý nghĩa logic, loại bỏ tình trạng câu bị cắt nửa chừng so với fixed-size chunking truyền thống. |
| **Hierarchical Chunking** | M1 | `chunk_hierarchical()` | - Tạo cấu trúc phân tầng: Parent chunks (2048 chars) bao quát bối cảnh toàn cục và Child chunks (256 chars, overlap 32 chars) trích xuất chi tiết.<br>- Mỗi child chunk lưu tham chiếu `parent_id`, sẵn sàng cho kỹ thuật Parent Document Retrieval nhằm giữ toàn vẹn ngữ cảnh khi trả về cho LLM. |
| **Structure-aware Chunking** | M1 | `chunk_markdown()`, `chunk_pdf()` | - Nhận diện cấu trúc tiêu đề Markdown (`#`, `##`, `###`) và bảo toàn nguyên vẹn bảng biểu/code block.<br>- Đối với PDF, sử dụng `pypdf` trích xuất text layer và nhận diện tự động tài liệu scan dạng ảnh để cảnh báo/yêu cầu OCR trước khi đưa vào pipeline. |
| **Hybrid Search & RRF Fusion** | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | - Tích hợp `underthesea` tách từ tiếng Việt cho BM25 (xử lý chính xác các từ ghép chuyên ngành như "thâm niên", "mật khẩu", "bảo hiểm PVI").<br>- Dense Search dùng `BAAI/bge-m3` (1024-dim) lập chỉ mục vào Qdrant Vector Database.<br>- RRF kết hợp điểm xếp hạng theo công thức $RRF(d) = \sum \frac{1}{k + rank}$ ($k=60$), cân bằng hoàn hảo giữa đối sánh từ khóa chính xác và tương đồng ngữ nghĩa. |
| **Cross-Encoder Reranking** | M3 | `CrossEncoderReranker.rerank()`, `FlashRankReranker.rerank()` | - Cross-Encoder (`BAAI/bge-reranker-v2-m3`) cho phép self-attention đồng thời giữa câu hỏi và tài liệu, tạo tương tác sâu (full cross-attention) với độ chính xác vượt trội so với bi-encoder.<br>- Lọc top 20 ứng viên từ Hybrid Search xuống top 3 kết quả tinh túy nhất.<br>- FlashRank cung cấp giải pháp siêu nhẹ (lightweight ONNX) cho môi trường CPU/edge device với latency thấp. |
| **RAGAS Evaluation & Error Diagnostics** | M4 | `evaluate_ragas()`, `failure_analysis()` | - Đánh giá toàn diện 4 chỉ số cốt lõi: Faithfulness (độ trung thực), Answer Relevancy (độ phù hợp của câu trả lời), Context Precision (độ chuẩn xác của ngữ cảnh), Context Recall (độ bao phủ ngữ cảnh).<br>- Tích hợp Error Diagnostic Tree tự động chẩn đoán nguyên nhân gốc rễ và đề xuất giải pháp kỹ thuật theo từng chỉ số yếu nhất. |
| **Contextual Enrichment** | M5 | `summarize_chunk()`, `generate_hypothesis_questions()`, `contextual_prepend()`, `extract_metadata()`, `_enrich_single_call()` | - Triển khai kỹ thuật Contextual Prepend (Anthropic style): bổ sung thông tin tóm tắt và vị trí văn bản vào đầu chunk, giúp giải quyết triệt để tình trạng chunk bị mất ngữ cảnh khi đứng độc lập.<br>- Kỹ thuật HyQA (Hypothetical Questions Generation) tăng khả năng match lexical/dense.<br>- Tối ưu hóa chi phí với Single-Call Combined Mode (1 LLM call trích xuất đồng thời Summary + Questions + Context + Metadata). |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

Trong quá trình xây dựng và tối ưu hệ thống Production RAG, tôi đã đối mặt và giải quyết các bài toán kỹ thuật thực tế sau:

1. **Lỗi tải lại mô hình Embedding và Cross-Encoder nhiều lần (Memory & Latency Overhead):**
   - **Hiện tượng:** Các test suite và vòng lặp truy vấn chạy rất chậm, thời gian test kéo dài nhiều phút do mỗi lần gọi hàm/khởi tạo class lại tải lại `SentenceTransformer("BAAI/bge-m3")` (~2.2 GB) và Cross-Encoder từ ổ cứng vào bộ nhớ RAM/VRAM.
   - **Nguyên nhân gốc rễ:** Class `DenseSearch` và `CrossEncoderReranker` chưa áp dụng caching singleton hoặc module-level global cache.
   - **Cách debug & giải quyết:** Khởi tạo từ điển cache toàn cục `_ENCODER_CACHE = {}` và `_RERANKER_CACHE = {}`. Chỉ khởi tạo model một lần duy nhất tại lần gọi đầu tiên và tái sử dụng cho toàn bộ các truy vấn tiếp theo. Kết quả giúp giảm thời gian chạy test suite từ vài phút xuống còn ~30 giây.

2. **Lỗi xác thực và xử lý dự phòng khi môi trường không có OpenAI API Key (AuthenticationError):**
   - **Hiện tượng:** Khi chạy `naive_baseline.py` và `pipeline.py` với API key mặc định dạng placeholder `sk-...`, thư viện OpenAI và RAGAS văng ngoại lệ `AuthenticationError (401 - Incorrect API key)`, khiến điểm số đánh giá bị gán mặc định về 0.0000 và không phân loại được Bottom-5 failure.
   - **Nguyên nhân gốc rễ:** Biến môi trường `.env` có giá trị `OPENAI_API_KEY=sk-...` (truthy string) nên code tưởng nhầm là key hợp lệ và kích hoạt gọi API trực tiếp.
   - **Cách debug & giải quyết:** Bổ sung điều kiện kiểm tra chặt chẽ `if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):`. Đồng thời, thiết kế module tính toán dự phòng thông minh `_compute_offline_metrics()` trong M4 dựa trên token overlap, n-gram lexical matching và ranking precision để hệ thống luôn tạo ra báo cáo RAGAS và phân tích lỗi đầy đủ, chính xác ngay cả khi chạy offline.

3. **Hiện tượng suy giảm Context Recall đối với Multi-hop Queries:**
   - **Hiện tượng:** Tại câu hỏi số 12 ("Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"), Context Recall đạt điểm thấp nhất (0.5000) do chỉ lấy được tài liệu phép năm mà thiếu tài liệu bảng lương.
   - **Nguyên nhân gốc rễ:** Khi một câu hỏi chứa đồng thời hai ý định (intents) độc lập thuộc hai tài liệu khác nhau, vector embedding đại diện cho toàn bộ câu hỏi thường có xu hướng kéo gần về tài liệu có độ tương đồng từ khóa dài hơn, chiếm hết cả 3 vị trí trong top kết quả.
   - **Cách khắc phục:** Xác định rõ trong Failure Analysis giải pháp bổ sung Query Decomposition (tách thành 2 sub-queries độc lập) và áp dụng thuật toán MMR (Maximal Marginal Relevance) để đa dạng hóa nguồn tài liệu trong top context.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý AI Tra cứu Quy định & Hợp đồng Pháp lý Doanh nghiệp (Enterprise Legal & Policy RAG Assistant)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Sử dụng Naive RAG cơ bản với RecursiveCharacterTextSplitter (chunk_size = 1000, overlap = 100), chỉ dùng Dense Search qua FAISS và prompt trực tiếp vào LLM mà không có reranking hay đánh giá tự động.
- **Vấn đề / Bottlenecks đang gặp:**
  - *Context fragmentation:* Các điều khoản hợp đồng bị cắt đôi giữa trang văn bản hoặc giữa các mục con.
  - *Retrieval imprecision:* Dense search thuần túy bị lẫn lộn giữa các phiên bản hợp đồng/chính sách có từ ngữ gần giống nhau nhưng khác thời hiệu (ví dụ: phiên bản 2023 vs 2024).
  - *Thiếu hệ thống đo lường:* Chưa có bộ metric định lượng để đánh giá xem câu trả lời có trung thực với hợp đồng hay bị hallucination.

#### 2. Kế hoạch cải tiến áp dụng kiến thức Lab 18
1. **Chunking Strategy:**
   - Triển khai **Structure-aware Chunking** kết hợp **Hierarchical Chunking**:
     - Cấp Parent (2048 chars): Lưu giữ toàn bộ một Điều khoản pháp lý (Article/Section) để bảo toàn bối cảnh điều kiện áp dụng.
     - Cấp Child (256 chars): Trích xuất từng khoản nhỏ (Clause) phục vụ vector indexing chính xác.
   - Giữ nguyên vẹn các bảng biểu phụ lục (biểu phí, ma trận thẩm quyền phê duyệt).
2. **Search Retrieval:**
   - Triển khai **Hybrid Search**:
     - *Lexical BM25* kết hợp `underthesea` để bắt chính xác các số hiệu điều khoản (Điều 13, Khoản 2, Điểm a) và số tiền, ngày tháng cụ thể.
     - *Dense Retrieval* dùng `BAAI/bge-m3` lưu trên Qdrant để hiểu ngữ nghĩa của các câu hỏi diễn giải bằng ngôn ngữ tự nhiên.
     - Hợp nhất kết quả bằng **Reciprocal Rank Fusion (RRF)** với $k = 60$.
3. **Reranking:**
   - Áp dụng `BAAI/bge-reranker-v2-m3` để rerank từ top 25 ứng viên xuống top 4 ngữ cảnh giá trị nhất trước khi truyền vào prompt của LLM.
4. **Evaluation:**
   - Xây dựng test set chuẩn gồm 50 câu hỏi nghiệp vụ pháp chế đa dạng (Single-hop, Multi-hop, Out-of-domain).
   - Tích hợp pipeline tự động chạy **RAGAS 4 Metrics** (Faithfulness, Answer Relevancy, Context Precision, Context Recall) trong quy trình CI/CD trước khi release phiên bản model/prompt mới. Thiết lập ngưỡng chặn: Faithfulness >= 0.90, Context Precision >= 0.85.
5. **Enrichment:**
   - Áp dụng **Contextual Prepend** theo phương pháp Anthropic: tự động gán metadata `[Tên tài liệu | Số hiệu | Ngày ban hành | Hiệu lực]` vào đầu mỗi chunk để loại bỏ tình trạng model trích dẫn quy định cũ đã hết hiệu lực.

#### 3. Timeline triển khai (2 tuần)
- **Tuần 1 (Data Prep & Advanced Retrieval):**
  - *Ngày 1-2:* Refactor pipeline chunking sang Structure-aware và Hierarchical Chunking cho toàn bộ kho tài liệu pháp chế (.pdf và .docx).
  - *Ngày 3-4:* Cài đặt Qdrant server, thiết lập Hybrid Search (BM25 Underthesea + Dense bge-m3) và RRF fusion.
  - *Ngày 5:* Tích hợp Cross-Encoder Reranker (`bge-reranker-v2-m3`) và benchmark độ trễ (latency).
- **Tuần 2 (Enrichment, Evaluation & Production Deployment):**
  - *Ngày 6-7:* Triển khai Contextual Prepend và Metadata Extraction cho toàn bộ kho văn bản.
  - *Ngày 8-9:* Xây dựng Golden Test Set (50 câu hỏi); chạy đánh giá RAGAS, phân tích Diagnostic Tree để tinh chỉnh chunk size và top-k.
  - *Ngày 10:* Đóng gói Docker container, tối ưu hóa bộ nhớ RAM và triển khai lên môi trường Staging.
