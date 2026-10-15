# Spec amendment v1.5 — Δ-CE (sau exp_0006)

**Quyết định của tác giả, 15/10/2026.** Thay E4 của v1.4. Dung sai H4 không đổi. Quyết định tương ứng: `research/DECISIONS.md` D-023.

```text
Spec amendment v1.5 — sau exp_0006, 15/10/2026. Thay E4 của v1.4. Dung sai H4 KHÔNG đổi.

F1. Pha 2 (SLSQP) chạy từ TẤT CẢ điểm xuất phát sau, khả thi hay không:
    z0, và điểm cuối pha 1 của mỗi restart (R = 5).
    Kết quả = nghiệm có d nhỏ nhất trong các nghiệm được XÁC NHẬN khả thi.
    Pha 1 giữ nguyên như v1.4 (R1, T = 400); vai trò của nó là khám phá điểm xuất phát
    cho bài toán không lồi, không còn là điều kiện cần để chạy SLSQP.
F2. SLSQP nhắm ràng buộc với κ' = κ + 1e-4. Khả thi được xác nhận bằng mô hình float32
    dùng trong đánh giá, với κ gốc. Áp dụng như nhau cho Δ-CE và CE thường (E2 giữ nguyên).
F3. H4 chạy trên hai bộ: bộ cũ (seed 0–999) và bộ mới (seed 1000–1999), mỗi bộ ở p ∈ {5, 20}.
    Phải đạt ở cả hai bộ.
F4. E4 thay bằng: H4 là kiểm tra kỹ thuật. Được sửa bộ giải để đạt H4, nhưng:
    (i)   không nới dung sai;
    (ii)  mỗi thay đổi thuật toán phải có lý do độc lập với các trường hợp cụ thể đã trượt,
          và được ghi vào DECISIONS;
    (iii) hạn chót cho H4 là hết ngày 17/10. Quá hạn mà chưa đạt → G1 trượt, quay lại tác giả.
F5. Bổ sung kiểm tra độ đầy đủ của bộ giải ở pilot G1 (validation Tier 0, chỉ để chẩn đoán):
    trên 50 điểm của cặp T1 seed 0, so sánh bộ giải chuẩn với SLSQP chạy từ 50 điểm xuất phát
    ngẫu nhiên trong hộp. Báo cáo tỉ lệ witness bộ giải chuẩn bỏ sót (trong ngân sách r)
    và chênh lệch khoảng cách. Không phải tiêu chí đạt/trượt; dùng để diễn giải found-rate.
```

## Lý do (tóm tắt từ tác giả)

- E4 đặt sai chỗ: chống fishing đúng cho kiểm định xác nhận trên dữ liệu thật (H1, H2), nhưng H4 là kiểm tra kỹ thuật trên dữ liệu tổng hợp, không có split test, không là bằng chứng cho claim nào. Điều cần bảo vệ là dung sai; (c) — nới tiêu chí — vẫn bị loại.
- SLSQP là SQP: xử lý được điểm xuất phát không khả thi; với trường hợp tuyến tính là QP lồi nên từ z0 giải chính xác. Ràng buộc "chỉ chạy từ ứng viên khả thi" ở E1 là thừa. Cách sửa có cơ sở lý thuyết, không phụ thuộc 33 cặp đã trượt — khác với "halve λ mỗi T/16" (chọn theo dữ liệu kiểm tra) và khởi tạo từ CE thường (không đảm bảo khả thi).
- F2: điểm nằm đúng trên biên ở float64 có thể không khả thi ở float32. F3: bộ seed mới là holdout cho chính kiểm tra kỹ thuật. F5: hỏi trực tiếp liệu bộ giải có bỏ sót witness trong ngân sách r trên dữ liệu thật.
