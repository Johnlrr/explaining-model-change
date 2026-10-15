# Spec amendment v1.2 — Δ-CE (SAU KHI THẤY KẾT QUẢ G0, exp_0001)

**Quyết định của tác giả, 15/10/2026.** G0 là gate dựng benchmark, không phải kiểm định chính; test chưa bị chạm; H1–H4 không bị ảnh hưởng. Các sửa đổi dưới đây là amendment, không tính vào giới hạn 2 lần chỉnh núm của §4.4. Quyết định tương ứng: `research/DECISIONS.md` D-018.

```text
B1. f_old: 30 epoch, Adam lr 1e-3 với cosine decay về 0. Giữ ngưỡng accuracy (≥ 0.95 Tier 0, ≥ 0.97 Tier 1).
    Không thay seed. Nếu sau B1 còn seed nào trượt ngưỡng → G0 trượt → quay lại tác giả (không tự sửa tiếp).
B2. Mọi tập adaptation (T1, T3, T4, N0, N1; cả Tier 1) mang cùng 5% nhiễu nhãn đối xứng như tập train.
    Sửa cách đọc D-017. N0 khác f_old chỉ ở thứ tự dữ liệu; mỗi drift khác chỉ ở chính drift đó.
B3. Thay tiêu chí "đồng ý" bằng "bất đồng dứt khoát":
    DD(f_old, f_new) = tỉ lệ điểm có argmax khác nhau VÀ |m_o| ≥ κ VÀ |m_n| ≥ κ (κ = 0.5, đã khóa).
    - N0, N1: DD ≤ 0.02 (thay agreement ≥ 0.98). Tiêu chí ΔFlip ≤ 0.05 giữ nguyên.
    - T3: DD ngoài nhóm con ≤ 0.05 (thay agreement ngoài nhóm con ≥ 0.95).
    - Áp dụng y hệt cho G0-Tier 1.
    Agreement thô vẫn được báo cáo (mô tả). B0 (descriptive H3) giữ nguyên định nghĩa thô.
B4. Bổ sung núm khai báo trước cho T3 và T4 (cả hai tier): lr fine-tune 3e-4 → 1e-3.
    Thứ tự dùng núm cho T3/T4: lần 1 = epoch 3 → 5 VÀ lr 3e-4 → 1e-3 (dùng cùng lúc);
    lần 2 = epoch 5 → 10. Vẫn tối đa 2 lần. Công thức fine-tune cuối cùng của từng điều kiện
    được báo cáo trong paper.
B5. Thứ tự chạy: exp_0002 = B1 + B2 + B3, T3/T4 dùng ngay núm lần 1 của B4
    (vì exp_0001 đã cho thấy 3 epoch ở lr 3e-4 không học được tương tác size × aspect).
    Nếu còn trượt thì exp_0003 dùng núm lần 2. Trượt nữa → Dừng hoặc quay lại tác giả.
```

## Lý do (tóm tắt từ tác giả)

- **B1–B2.** Với 5% nhiễu nhãn đối xứng, cross-entropy thấp nhất ≈ H(0.05) ≈ 0.199. f_old dừng ở ≈ 0.28 (chưa hội tụ), còn fine-tune N0 xuống 0.086, thấp hơn mức sàn, chỉ đạt được khi nhãn sạch. Vậy N0 trong exp_0001 là "học lại trên nhãn sạch".
- **B3 thay cho đề xuất c.3 (|φ₁ − 0.5| ≥ 0.05).** c.3 không chuyển được sang Tier 1 (không có ngưỡng φ₁), dùng ground truth của factor nhiệm vụ để lọc, và không khớp với L1 (witness chỉ tồn tại nơi |h| ≥ 2κ). DD dùng κ đã khóa, áp dụng cho cả hai tier, và chính là F_κ (loại N ∪ O) quan sát được trên dữ liệu; rung ngưỡng tự bị loại vì sát ngưỡng ít nhất một mô hình có margin < κ.
- **B4–B5.** Loss fine-tune T3 giảm chậm (0.72 → 0.58 → 0.53 ở seed 0); lr 3·10⁻⁴ quá nhỏ cho một tương tác mới trên 25% dữ liệu bị đảo nhãn. T4 seed 2 (0.34) cũng train chưa đủ: dải bị dịch chỉ ≈ 9% dữ liệu.
- **Công thức fine-tune khác nhau giữa các điều kiện** (N0/N1: 3 epoch, lr 3·10⁻⁴; T3/T4 mạnh hơn) là đặc điểm của từng drift. Paper nêu rõ. Ảnh hưởng lên H1/H2 nhỏ, vì các phương pháp được so trên cùng một cặp mô hình.
