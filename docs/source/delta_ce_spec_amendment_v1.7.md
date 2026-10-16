# Spec amendment v1.7 — Δ-CE (SAU KHI THẤY KẾT QUẢ G2, exp_0012 / exp_0013)

**Quyết định của tác giả, 16/10/2026**, theo đề xuất của Research Lead. Đây là thay đổi **sau khi đã thấy kết quả G2 trên validation** (test chưa bị chạm). Quyết định tương ứng: `research/DECISIONS.md` D-030. Quy tắc quyết định dưới đây được khai báo **trước** khi chạy chẩn đoán D1–D3.

```text
Spec amendment v1.7 — 16/10/2026, SAU KHI THẤY KẾT QUẢ G2 (exp_0013). Test chưa bị chạm.

H1. f_old Tier 1: 60 epoch, Adam lr 1e-3, cosine decay về 0 (số bước ≈ Tier 0: 20 000 × 30 / 256 ≈ 2 340 bước
    tương đương ≈ 56 epoch trên 10 612 ảnh). Ngưỡng accuracy trên validation: 0.97 → 0.95 (bằng Tier 0).
    Đây là thay đổi ngưỡng gate sau khi thấy kết quả. Seed nào < 0.95 → quay lại tác giả.
H2. Chẩn đoán (CPU, validation, không chạm test, không đọc Recovery):
    D1  accuracy của f_old (seed 0, 60 epoch) trên ảnh thật và trên D(μ(x)) của validation, so với nhãn sạch;
    D2  linear probe (hồi quy logistic, L2 = 1e-3, chuẩn hóa theo trung bình/độ lệch chuẩn của train) huấn luyện
        trên μ(x) của train, accuracy trên μ(x) của validation; lưới 32 ảnh gốc cạnh ảnh tái tạo;
    D3  accuracy của f_old và shape oracle (exp_0012) theo mức ladder (mọi mức hợp lệ, màu giữ nguyên)
        và theo bin độ dày [1.5, 2, 2.5, 3, 3.5, 4, 5] của mức được chọn.
H3. Quy tắc quyết định cho VAE và shape oracle:
    (a) probe ≥ 0.95 VÀ accuracy f_old trên D(μ(x)) ≥ 0.90: giữ VAE; train lại shape oracle trên α̂ của ảnh thật
        (mọi mức ladder hợp lệ của train) cộng α̂ của D(μ(x)) (một ảnh tái tạo cho mỗi ảnh train), nhãn là chữ số gốc.
    (b) ngược lại: thử VAE β ∈ {1, 0.5, 0.1} trên CPU, 30 epoch mỗi mức, latent 16. Chọn β LỚN NHẤT thỏa đồng thời
        accuracy f_old (seed 0, 60 epoch) trên D(μ(x)) validation ≥ 0.95 và ba tiêu chí Spearman của oracle (§9 G2).
        Không β nào thỏa → quay lại tác giả. Sau đó train shape oracle trên dữ liệu như nhánh (a) với VAE đã chọn.
    Shape oracle (cả hai nhánh): Adam 1e-3, cosine về 0, 20 epoch trên tập ghép; tiêu chí giữ nguyên
    (≥ 0.98 ảnh thật, ≥ 0.95 ảnh tái tạo). Không đạt → quay lại tác giả.
H4. T3 Tier 1: trong cùng một job train bốn biến thể trên f_old mới: mặc định (3 epoch, lr 3e-4); núm 1 (5 epoch,
    lr 1e-3); núm 2 (10 epoch, lr 1e-3); dự phòng bg > q85 (k* = bg, công thức mặc định). Đánh giá theo thứ tự đó,
    lấy biến thể ĐẦU TIÊN đạt C1; núm 1 chỉ tính là một lần chỉnh khi mặc định trượt, núm 2 khi núm 1 trượt;
    dự phòng chỉ khi cả hai núm trượt. Không biến thể nào đạt → quay lại tác giả. Quantile 0.80 không dùng.
H5. Mọi cặp f_new Tier 1 (T1, T3, N0) train lại trên f_old mới; các tiêu chí G0 T1/N0 đánh giá lại. Một job (2/4).
H6. Hạn 22/10. Nếu Tier 1 vẫn trượt G2 sau vòng sửa này, Tier 1 bị loại khỏi acceptance:
    H1 và H2b không được kiểm định; Route A = H4 ∧ H2a, Holm trên họ 1 kiểm định; không có claim về bộ giải
    (quy tắc B5 cần cả hai tier). Paper chỉ còn dữ liệu tổng hợp.
H7. Thiết bị train Tier 1 (Kaggle T4 hay CPU local) do tác giả chọn sau khi có ước tính giờ CPU (sửa D-027 G2 nếu chọn local).
```

## Lý do (tóm tắt)

- 420 bước của 10 epoch là quá ít (chẩn đoán trên validation: 0.911 / 0.960 / 0.9635 ở 10 / 30 / 60 epoch). Số bước tương đương Tier 0 là lý do độc lập với kết quả cụ thể; ngưỡng hạ xuống bằng Tier 0 vì nó chỉ bảo đảm classifier đủ tốt, không làm lệch so sánh giữa các phương pháp.
- Shape oracle 0.551 trên ảnh tái tạo là vấn đề chặn đường: điểm đánh giá, witness và Recovery đều nằm trên D(z). Cần biết ảnh tái tạo còn giữ hình dạng không (D1, D2) trước khi chọn sửa oracle hay VAE.
- Khai báo trước quy tắc quyết định để việc chọn nhánh không phụ thuộc vào ý muốn sau khi thấy số liệu.
