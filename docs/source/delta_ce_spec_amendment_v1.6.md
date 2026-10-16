# Spec amendment v1.6 — Δ-CE (sau pilot G1, exp_0009)

**Quyết định của tác giả, 16/10/2026.** Sau pilot G1, **trước** khi đo tốc độ của (A). Không đổi thuật toán, không đổi maxiter; chỉ đổi tiêu chí khả thi về runtime và môi trường chạy. Quyết định tương ứng: `research/DECISIONS.md` D-027.

```text
Spec amendment v1.6 — 16/10/2026, sau pilot G1 (exp_0009), TRƯỚC khi đo tốc độ của (A).
Không đổi thuật toán, không đổi maxiter; chỉ đổi tiêu chí khả thi về runtime và môi trường chạy.

G1. Bỏ tiêu chí "runtime ≤ 5 phút mỗi cặp". Thay bằng tiêu chí dự phóng
    (tính từ runtime đo được sau (A) × số điểm × số cặp × số phương pháp):
    (i)  Tier 0, toàn bộ lượt chạy cuối: ≤ 36 giờ wall-clock trên máy local;
    (ii) Tier 1, phần đánh giá: ≤ 36 giờ wall-clock trên máy local;
    (iii) GPU-hours tích lũy ≤ 10 (giữ nguyên, CONSTRAINTS §2).
    Đạt cả ba → G1 qua phần runtime.
G2. Môi trường: Kaggle T4 chỉ dùng để train (VAE, shape oracle, f_old, f_new ở Tier 1).
    Mọi bộ giải và baseline của Tier 1 chạy local trên CPU, tải checkpoint về và kiểm hash.
    Sửa D-006 và D-010 tương ứng. Mọi phương pháp chạy trên cùng thiết bị.
G3. Runtime là metric phụ: báo cáo wall-clock và CPU-time mỗi điểm cho từng phương pháp,
    TÁCH theo điểm có và không có witness, kèm ghi chú rằng chi phí của Δ-CE
    tập trung ở việc kết luận "không có witness trong ngân sách".
G4. Phương án (B), dừng sớm cho SLSQP, CHỈ được xét nếu G1 trượt sau (A). Khi đó:
    - quy tắc phải có lý do độc lập với số liệu pilot, ghi vào DECISIONS trước khi chạy;
    - chỉ được chấp nhận nếu, trên các cặp pilot, found-rate từng cặp và tập ID điểm
      tìm được witness TRÙNG KHỚP với bộ giải hiện tại (không đọc vector attribution).
G5. (A) phải cho kết quả trùng bit với bộ giải hiện tại trên các cặp pilot.
    Nếu không trùng bit thì phải chỉ ra được nguyên nhân thuần số học, và tập ID điểm tìm được witness không đổi.
```

## Lý do (tóm tắt từ tác giả)

- Chi phí của Δ-CE dồn vào những điểm không có witness: SLSQP phải chạy hết maxiter mới kết luận "không tìm thấy". Đó là cái giá của chính hành vi null mong muốn; B1 rẻ hơn vì luôn trả về một đáp án, kể cả trên cặp null. Paper báo cáo runtime theo hướng này.
- "≤ 5 phút mỗi cặp" là ước lượng thay thế cho ràng buộc lịch và ngân sách, không có ý nghĩa khoa học; đổi nó không làm lệch so sánh giữa các phương pháp. Thay bằng chính các ràng buộc nó đại diện.
- SLSQP là việc của CPU: Kaggle GPU chỉ để train; đánh giá Tier 1 chạy local, mọi phương pháp cùng thiết bị.
