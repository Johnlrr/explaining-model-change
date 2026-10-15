# Spec amendment v1.3 — Δ-CE (SAU KHI THẤY KẾT QUẢ G0, exp_0002 và exp_0003)

**Quyết định của tác giả, 15/10/2026.** Đây là lần thay đổi tiêu chí G0 cuối cùng; không có amendment nào khác cho G0 sau mốc này. Test chưa bị chạm; H1–H4 không bị ảnh hưởng. Quyết định tương ứng: `research/DECISIONS.md` D-020.

```text
Spec amendment v1.3 — SAU KHI THẤY KẾT QUẢ G0 (exp_0002, exp_0003), 15/10/2026.
Đây là lần thay đổi tiêu chí G0 cuối cùng. Không có amendment nào khác cho G0 sau mốc này.
Test chưa bị chạm; H1–H4 không bị ảnh hưởng.

C1. Tiêu chí G0 cho T3 (thay "accuracy nhóm con ≥ 0.85" và "đồng ý ngoài nhóm con"):
    (i)   ΔFlip_{k*} ≥ 0.15, với k* = size (Tier 0) / thick (Tier 1);
    (ii)  max_{k ≠ k*} |ΔFlip_k| ≤ 0.05  (thay đổi gán được độc quyền cho k*);
    (iii) DD ngoài nhóm con ≤ 0.05 (giữ nguyên từ v1.2).
    Cả ba điều kiện, ở mọi seed.
C2. Tiêu chí cũ (accuracy nhóm con theo nhãn mới, cả bản thô lẫn bản chỉ tính điểm dứt khoát)
    vẫn được tính và BÁO CÁO trong paper cùng với lý do thay đổi; không được lược bỏ.
    Paper nêu hạn chế: "biên nhóm con mà f_new học được là mềm và rò xuống dưới ngưỡng tiêm vào,
    làm thay đổi T3 bớt phi cục bộ so với thiết kế; điều này bất lợi cho Δ-CE trong H2".
C3. Tier 1 T3 (thick > q85): dùng C1 NGAY TỪ ĐẦU. Đây là khai báo trước, vì Tier 1 chưa chạy.
    Núm của §4.4 (epoch; quantile 0.85 → 0.80) cùng núm lr ở v1.2/B4 vẫn áp dụng, tối đa 2 lần.
    Nếu vẫn trượt → dùng phương án dự phòng đã có ở G2: T3-Tier1 = đảo nhãn theo bg > q85,
    với k* = bg, cùng tiêu chí C1.
    Nếu phương án đó cũng trượt → bỏ T3-Tier1. Khi đó H2b không được kiểm định
    và H1 chỉ tính trên T1-Tier1. Holm áp dụng trên các kiểm định còn lại.
C4. T1, T4, N0, N1: tiêu chí giữ nguyên như v1.2.
```

## Lý do (tóm tắt từ tác giả)

- **Ground truth không đổi.** Recovery@r chấm việc phương pháp chỉ ra *size* là factor thay đổi. Chẩn đoán exp_0003: ΔFlip_size ≈ +0.32 ở cả 3 seed, các factor khác nhỏ. Biên mềm đổi khoảng cách tới witness, không đổi k*. Ngưỡng accuracy 0.85 đặt tùy ý và không đo đúng điều G0 cần xác nhận.
- **Biên mềm bất lợi cho Δ-CE.** Thay đổi rò xuống vùng size 0.70–0.75 với margin thấp, nên bớt phi cục bộ; gradient (B3) bắt hướng size dễ hơn, H2 khó đạt hơn so với thiết kế gốc.
- **Test chưa bị chạm.** Thay đổi chỉ quyết định T3 có vào benchmark hay không.
- **(B) bỏ T3** làm mất H2a và H2b, tức thí nghiệm phân biệt Δ-CE với xấp xỉ gradient.
- **Điều kiện (ii) độc quyền** được thêm so với đề xuất của Research Lead (chỉ ΔFlip_size ≥ 0.15), vì Recovery@r chỉ có nghĩa khi thay đổi nằm trên một factor.

## Triển khai

- Không chạy lại T3-Tier 0. G0 được tính lại trên các checkpoint đã lưu (exp_0002: f_old, N0, N1; exp_0003: T1, T3, T4), hash đã kiểm tra, trong run `exp_0004` (chỉ đánh giá lại, không train).
