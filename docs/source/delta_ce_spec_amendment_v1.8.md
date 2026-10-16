# Spec amendment v1.8 — Δ-CE (G2 vòng hai, SAU KHI THẤY exp_0013–exp_0015)

**Quyết định của tác giả, 16/10/2026**, theo đề xuất của Research Lead. Thay đổi thiết kế benchmark Tier 1 **sau khi thấy kết quả G2 trên validation**; test chưa bị chạm, không có Recovery nào được đọc. Đây là **vòng sửa cuối cùng** của G2. Quyết định tương ứng: `research/DECISIONS.md` D-031. Mọi quy tắc dưới đây được khai báo **trước** khi chạy vòng hai.

```text
Spec amendment v1.8 — 16/10/2026, G2 vòng hai (cuối cùng). Test chưa bị chạm.

I1. Dữ liệu Tier 1 v2 (thay §4.3 về độ dày): bỏ erosion. Ladder chỉ dùng các mức {0, +1, +2, +3};
    t ~ U[2.5, 6.5], chọn mức hợp lệ có độ dày đo được gần t nhất. Độ dày đo được vẫn là ground truth.
    Dẫn xuất tất định từ tier1_data.npz (exp_0010, sha256 e74fe388…): giữ nguyên ảnh, nhãn, split, φc, φb;
    lấy mẫu lại t với seed 2235 / 2236 / 2237 (train / val / test). Hash của file dẫn xuất ghi vào manifest.
    Lý do độc lập với kết quả: sàn đo 2.0 và mức −2 chỉ hợp lệ cho 51% ảnh (NOTES, exp_0010, trước G2).
I2. VAE: 100 epoch, latent 16, β ∈ {0.1, 0.05, 0.02}. Shape oracle (mỗi β): train trên α̂ của ảnh thật
    (mọi mức hợp lệ của train) cộng α̂ của D(μ(x)) của train, nhãn là chữ số gốc; Adam 1e-3 cosine, 20 epoch.
    Chọn β LỚN NHẤT thỏa đồng thời:
      (i)   tiêu chí G2 của spec: Spearman ≥ 0.9 / 0.9 / 0.8 (color / bg / thick) trên D(μ(x));
            shape oracle ≥ 0.98 trên ảnh thật và ≥ 0.95 trên ảnh tái tạo (validation);
      (ii)  valid_rate_val × 1 991 ≥ 300 cho MỌI cặp (T1 và N0, cùng biến thể T3 được chọn nếu có);
            valid_rate = tỉ lệ ảnh validation có m_o(D(μ(x))) ≥ κ và m_n(D(μ(x))) ≥ κ với nhãn sạch (spec §3.2);
            1 991 = số ảnh test 4/9; không dùng gì khác từ test;
      (iii) accuracy f_old (seed 0) trên D(μ(x)) validation ≥ 0.90 — ngưỡng TỰ ĐẶT, không phải tiêu chí spec,
            giữ lại để hạn chế hiệu ứng chọn mẫu (điểm hợp lệ thiên về ảnh VAE giữ đúng lớp).
    Tiêu chí "f_old trên D(μ(x)) ≥ 0.95" của v1.7 H3 bị thay.
I3. f_old: 60 epoch cosine, ngưỡng 0.95 (v1.7 H1), không đổi dù kết quả cao hơn.
I4. Biến thể f_new (một lượt chạy): T1 mặc định → núm 1 (tương quan 0.99) → núm 2 (tương quan 0.99, 5 epoch);
    T3 mặc định (3 epoch, lr 3e-4) → núm 1 (5 epoch, lr 1e-3) → núm 2 (10 epoch, lr 1e-3) → dự phòng bg > q85
    (k* = bg, công thức mặc định). Mỗi điều kiện lấy biến thể ĐẦU TIÊN đạt G0 ở cả 3 seed. N0 không có núm.
I5. Thiết bị: toàn bộ Tier 1 train trên CPU local (sửa D-027 / v1.6 G2). Không dùng Kaggle job nào.
I6. Quy tắc trượt hai tầng:
    Tầng Tier 1: (một β thỏa I2) ∧ (mọi f_old ≥ 0.95) ∧ (G0 đạt ở T1 và N0).
      Thiếu → loại Tier 1 (v1.7 H6): Route A = H4 ∧ H2a, Holm m = 1, không claim về B5.
    Tầng T3-Tier 1: một biến thể T3 đạt C1. Không biến thể nào đạt → chỉ bỏ T3-Tier 1 (v1.3 C3):
      H2b không kiểm định; H1 chỉ trên T1-Tier 1; Holm m = 2 (H1, H2a); không claim về B5.
      Biến thể bg đạt → H2b kiểm định với k* = bg, ghi rõ trong paper.
    Vòng hai trượt ở tầng Tier 1 → không có vòng ba.
I7. Báo cáo: phân bố độ dày mới và q85; std của μ(x) (thang đo latent, ảnh hưởng Reg và r); số điểm hợp lệ trên
    validation của từng cặp; accuracy shape oracle trên ảnh tái tạo cho mỗi β.
```
