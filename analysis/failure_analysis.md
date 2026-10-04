# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Phạm Đình Bảo Khôi  
**Khóa:** K4 - Track 3A  

---

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8333 | 0.8591 | +0.0258 |
| Answer Relevancy | 0.5176 | 0.6465 | +0.1289 |
| Context Precision | 0.9333 | 0.9917 | +0.0583 |
| Context Recall | 0.9250 | 0.8750 | -0.0500 |

## Bottom-5 Failures

### #1
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** (Dựa trên context được cung cấp)
- **Got:** Phạt 50.000 VNĐ.
- **Worst metric:** faithfulness (0.1818)
- **Error Tree:** Output sai → Context đúng? (Có thể) → Query OK? → Root cause: LLM tự tính toán và giải thích dài dòng dẫn đến tính sai công thức phạt hoặc context không nói chính xác là 50.000 VNĐ (hallucination).
- **Suggested fix:** Cải thiện Prompt để LLM không tự tính toán quá sâu mà chỉ trích xuất mức phạt/cách phạt trong ngữ cảnh. Hoặc thêm các ví dụ Few-shot.

### #2
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** (Dựa trên context được cung cấp)
- **Got:** Nhân viên có thâm niên từ 3 năm trở lên sẽ được cộng thêm 1 ngày phép...
- **Worst metric:** answer_relevancy (0.2670)
- **Error Tree:** Output sai (có thể đúng) → Context đúng? → Root cause: Do query ngắn, câu trả lời tuy đúng context nhưng cách diễn đạt không khớp hoàn toàn với embedding của câu hỏi khiến mô hình đánh giá Relevancy thấp.
- **Suggested fix:** Áp dụng query expansion để câu hỏi gốc rõ nghĩa hơn.

### #3
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** (Dựa trên context được cung cấp)
- **Got:** Nghỉ phép không lương 20 ngày cần phê duyệt của Giám đốc điều hành (CEO).
- **Worst metric:** answer_relevancy (0.2733)
- **Error Tree:** Output sai (do relevancy bị chấm thấp). → Root cause: Câu trả lời rất ngắn và dứt khoát, nhưng có thể RAGAS chấm điểm Answer Relevancy kém do context có thể chứa các điều khoản phức tạp hơn nhưng không được đề cập.
- **Suggested fix:** Hướng dẫn LLM trả lời kèm diễn giải đầy đủ về quy trình thay vì chỉ nêu đích danh người phê duyệt.

### #4
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** (Dựa trên context được cung cấp)
- **Got:** Nhân viên phải hoàn trả 100% chi phí. Phải hoàn trả 25.000.000 VNĐ.
- **Worst metric:** faithfulness (0.3333)
- **Error Tree:** Output sai → Context đúng? → Root cause: Context có thể nói nếu làm dưới 12 tháng thì đền bù một tỷ lệ khác hoặc 100%. Nếu LLM lấy sai tỉ lệ thì Faithfulness sẽ bị kéo xuống. Tính toán suy luận thời gian của LLM yếu.
- **Suggested fix:** Chuyển câu hỏi dạng toán đố sang cho tool/agents xử lý hoặc bắt LLM trích xuất luật lệ trước rồi mới giải bài.

### #5
- **Question:** Phụ cấp ăn trưa hàng tháng là bao nhiêu?
- **Expected:** (Dựa trên context được cung cấp)
- **Got:** Phụ cấp ăn trưa hàng tháng là 1.000.000 VNĐ/tháng.
- **Worst metric:** answer_relevancy (0.4095)
- **Error Tree:** Output đúng nhưng Answer Relevancy bị chấm kém. → Root cause: Do format câu trả lời quá ngắn gọn, RAGAS sinh lại câu hỏi từ câu trả lời này và thấy không khớp hoàn toàn với câu hỏi gốc.
- **Suggested fix:** Ép System Prompt trả lời đầy đủ chủ vị và lặp lại một phần câu hỏi.
