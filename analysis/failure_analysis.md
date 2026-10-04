# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Minh Tuấn  
**Mã số học viên:** 2A202602420  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.8576 | 0.8159 | -0.0417 |
| Context Precision | 0.9917 | 0.9917 | +0.0000 |
| Context Recall | 0.8592 | 0.7902 | -0.0690 |

---

## Bottom-5 Failures

### #1
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Trích xuất các điều khoản về hạn mức mua sắm và thủ tục hóa đơn tài chính thông thường từ quy chế mua sắm.
- **Worst metric:** `context_recall` (score: 0.6154)
- **Error Tree:** Output thiếu thông tin phòng CNTT → Context chỉ bao gồm hạn mức tiền mà thiếu điều kiện thẩm định CNTT → Query phức hợp ("ai phê duyệt" VÀ "cần gì từ phòng CNTT")
- **Root cause:** Câu hỏi truy vấn kết hợp hai khía cạnh thông tin (thẩm quyền phê duyệt ngân sách và quy chuẩn kỹ thuật thiết bị). Do chia chunk theo child chunk nhỏ (256 chars), phần thẩm định kỹ thuật bị tách riêng khỏi bảng thẩm quyền ngân sách chung.
- **Suggested fix:** Áp dụng Parent-Document Retrieval (trả về parent chunk 2048 chars cho LLM khi child chunk 256 chars match), hoặc mở rộng cửa sổ ngữ cảnh (Context Window Expansion) để giữ trọn vẹn quy trình mua sắm CNTT.

### #2
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Thiết bị trên 50 triệu VNĐ thuộc thẩm quyền phê duyệt của Tổng Giám đốc (CEO) và cần có ý kiến thẩm định của Giám đốc Tài chính (CFO) trước khi trình ký.
- **Got:** Thông tin hạn mức phê duyệt chung từ 5-50 triệu và trên 50 triệu nhưng thiếu phần ghi chú về ý kiến CFO.
- **Worst metric:** `context_recall` (score: 0.5385)
- **Error Tree:** Output nêu được cấp CEO nhưng thiếu vai trò của CFO → Context chỉ lấy được 1 dòng bảng → Cắt bảng biểu chưa tối ưu
- **Root cause:** Tài liệu `mua_sam.md` lưu trữ phân cấp thẩm quyền dưới dạng bảng Markdown. Khi chunking theo kích thước token cố định, phần chú thích chân bảng (CFO thẩm định) bị ngắt sang chunk tiếp theo và bị reranker lọc mất khỏi top-3.
- **Suggested fix:** Áp dụng Structure-aware Chunking để nhận diện các khối bảng biểu Markdown và giữ nguyên toàn bộ bảng cùng các ghi chú liên quan trong một chunk duy nhất.

### #3
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9 ÷ 3 = 3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Trích xuất thông tin về quy chế phép năm v2024 (15 ngày cơ bản + 1 ngày cho mỗi 3 năm thâm niên), thiếu hoàn toàn thông tin dải lương Senior.
- **Worst metric:** `context_recall` (score: 0.5000)
- **Error Tree:** Output chỉ trả lời được vế phép năm → Context hoàn toàn thiếu tài liệu `bang_luong_2024.md` → Query là Multi-hop truy vấn 2 thực thể độc lập
- **Root cause:** Đây là câu hỏi dạng Multi-hop / Compound query đòi hỏi dữ liệu từ 2 file riêng biệt (`nghi_phep_nam_v2024.md` và `bang_luong_2024.md`). Cơ chế Dense Search và Cross-encoder có xu hướng tính score cao cho một chủ đề chiếm ưu thế trong câu hỏi, khiến cả top-3 kết quả đều thuộc tài liệu phép năm.
- **Suggested fix:** Tích hợp Query Decomposition: phân rã câu hỏi thành 2 câu hỏi con (Sub-query 1: "Nhân viên Senior 9 năm thâm niên được bao nhiêu ngày phép năm?", Sub-query 2: "Dải lương của nhân viên Senior là bao nhiêu?"), retrieve song song rồi dùng Reciprocal Rank Fusion kết hợp ngữ cảnh.

### #4
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ phép không lương trên 14 ngày (từ 15 ngày trở lên) cần có sự phê duyệt của Trưởng bộ phận, Giám đốc Nhân sự (HRD) và Tổng Giám đốc (CEO).
- **Got:** Trích xuất đoạn quy định chung về điều kiện nghỉ không lương và các trường hợp nghỉ dưới 14 ngày.
- **Worst metric:** `context_recall` (score: 0.5833)
- **Error Tree:** Output nêu chưa chính xác các cấp ký duyệt cho mốc 20 ngày → Context thiếu đoạn phân cấp theo thời gian nghỉ → Ranh giới chunking chia cắt quy định
- **Root cause:** Quy định nghỉ không lương chia làm nhiều bậc thời gian (1-3 ngày, 4-14 ngày, >=15 ngày). Việc phân đoạn chunking ngắt ngang giữa các đề mục khiến điều khoản về mốc thời gian dài hạn bị xếp vào chunk sau và có score lexical thấp hơn phần mở đầu.
- **Suggested fix:** Cải tiến Semantic chunking hoặc Structure-aware chunking dựa trên Header Markdown (`### Quy định theo thời hạn nghỉ`) để gom nhóm toàn bộ ma trận thẩm quyền vào cùng một đơn vị lưu trữ.

### #5
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Thông tin bảng lương, thu nhập của nhân viên thuộc cấp độ "Tối mật" (Confidential / Cấp độ 3), tuyệt đối không chia sẻ ra ngoài bộ phận C&B và Ban Giám đốc.
- **Got:** Trích xuất bảng định nghĩa các cấp độ phân loại dữ liệu (Công khai, Nội bộ, Bảo mật, Tối mật) nhưng thiếu ví dụ áp dụng cụ thể cho bảng lương.
- **Worst metric:** `context_recall` (score: 0.6552)
- **Error Tree:** Output liệt kê các cấp độ dữ liệu nhưng chưa gắn trực tiếp với bảng lương → Context trích xuất từ tài liệu quản trị dữ liệu chưa liên kết với tài liệu lương → Ambiguity giữa các file
- **Root cause:** Từ khóa "lương" xuất hiện trong nhiều tài liệu (`ky_luong.md`, `bang_luong_2024.md`, `phan_loai_du_lieu.md`). Khi tìm kiếm, thuật toán bị nhiễu do tài liệu bảng lương có tần suất từ khóa cao hơn tài liệu phân loại dữ liệu.
- **Suggested fix:** Bổ sung Contextual Prepend (M5) gắn nhãn rõ ràng nguồn tài liệu `[Tài liệu Phân loại & Bảo vệ dữ liệu]` trước từng chunk, kết hợp Metadata Filtering theo `category="security_policy"`.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
> *"Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"*

### Error Tree walkthrough:
1. **Output đúng?** → **Chưa hoàn chỉnh (Partial Failure).**  
   Hệ thống chỉ trích xuất và tính toán được số ngày phép năm (18 ngày: 15 ngày cơ bản + 3 ngày thâm niên), hoàn toàn bỏ sót dải lương của vị trí Senior (20 - 35 triệu VNĐ/tháng).
2. **Context đúng?** → **Chưa đúng và thiếu.**  
   Top 3 context trả về bởi Reranker chỉ toàn các đoạn văn từ `nghi_phep_nam_v2024.md`. Không có bất kỳ chunk nào từ `bang_luong_2024.md` lọt vào top 3 để cung cấp thông tin cho bước Generation.
3. **Query rewrite OK?** → **Chưa có bước Query Transformation.**  
   Câu hỏi gốc là câu hỏi kép (Compound Query) chứa 2 intent độc lập. Hiện tại Pipeline gửi nguyên vẹn câu hỏi này vào Hybrid Search, dẫn đến vector embedding bị thiên lệch về ngữ nghĩa của vế dài hơn ("nghỉ phép năm, thâm niên").
4. **Fix ở bước:**  
   - **Bước 1 (Pre-retrieval):** Bổ sung module **Query Decomposition** (phân rã thành 2 sub-queries).
   - **Bước 2 (Post-retrieval / Reranking):** Bổ sung cơ chế **Diversity Reranking (Maximal Marginal Relevance - MMR)** nhằm phạt độ tương đồng dư thừa giữa các context đã chọn, đảm bảo mỗi chủ đề trong câu hỏi đều có ít nhất 1 chunk đại diện trong top kết quả.

### Nếu có thêm 1 giờ, sẽ optimize:
1. **Triển khai Query Decomposition Agent:** Dùng LLM nhẹ (hoặc regex pattern) để tự động phát hiện liên từ "và", "đồng thời", "song song" nhằm tách compound query thành danh sách single queries, sau đó chạy multi-query retrieval song song.
2. **Kích hoạt Parent Document Retriever:** Lưu trữ child chunk (256 tokens) để embedding và semantic search chính xác, nhưng khi tổng hợp context cho LLM thì truy xuất parent chunk (2048 tokens) chứa toàn bộ phần bối cảnh xung quanh.
3. **Thiết lập Hybrid Reranking với MMR:** Thay vì chỉ sắp xếp thuần túy theo điểm Cross-encoder, kết hợp MMR với hệ số $\lambda = 0.7$ để vừa tối ưu độ liên quan vừa đa dạng hóa nguồn tài liệu trong top context.
