# Spec amendment v1.1 — Δ-CE (trước freeze 17/10/2026)

**Quyết định của tác giả, 15/10/2026.** Sửa `delta_ce_spec.md` v1.0. Ở mọi điểm khác nhau, amendment này thay thế spec v1.0. Các mục `docs/` được cập nhật theo đây; quyết định tương ứng ghi ở `research/DECISIONS.md` (D-011 đến D-015).

```text
A1. Split: G0, G1-pilot, G2, σ_k dùng validation (Tier 1: toàn bộ validation cho G0).
    Test chỉ dùng ở final procedure. D-009 bị thay bằng D-011; quay lại tinh thần D-002.
    EVALUATION §6: bỏ bảng ngoại lệ test; các hạn chế trên pilot vẫn giữ nhưng áp cho validation.
A2. H3 → descriptive. Thêm B0 = tỉ lệ bất đồng Monte Carlo (Hammer & Hüllermeier) trên điểm đánh giá.
    Báo cáo AUROC(S) của Δ-CE, B0, B1, B3; ghi "conditional on G0-qualified pairs".
A3. Acceptance (EVALUATION §15): Route A = H4 ∧ (H1 ∨ H2a ∨ H2b), Holm như cũ.
    Route B = H4 ∧ ¬(H1 ∨ H2a ∨ H2b) → paper benchmark + phân tích. Bỏ B′ (gộp vào B).
A4. Baseline mới (không vào Holm, báo cáo trong bảng chính):
    B5 (DIG-style): line search dọc đoạn z0 → K=20 điểm train gần nhất (cùng không gian Z,
       metric d), trả về điểm thỏa F^N_κ có d nhỏ nhất ≤ r; 50 bước/đoạn.
    A1 đổi tên "DeepXplore-style" (λ=0), báo cáo cạnh B5.
    Khai báo trước: nếu |Recovery(Δ-CE) − Recovery(B5)| ≤ 0.10 trên T3 cả hai tier,
    claim chuyển sang lý thuyết + benchmark, không claim solver.
A5. Định vị: Δ-CE trình bày như thích nghi từ DIG (L-022) và DeepXplore (L-023);
    P4 trích Artelt Thm 1, không claim. Đóng góp chính: (1) benchmark ground-truth,
    (2) so sánh có kiểm soát các họ giải thích + P-tie, (3) witness margin-κ với L1/P2/P3.
A6. NPL 2022: chấp nhận rủi ro còn lại; thử trang tác giả trước 17/10, không chặn G0.
```

## Lý do (tóm tắt từ tác giả)

- **A1.** Dùng test cho gate là lỗi khi viết spec, không phải ý đồ thiết kế. Chưa chạy gì nên sửa không tốn gì, và paper khỏi phải mang caveat về tính blind của test. Tier 1 validation 4/9 có khoảng 1 200 ảnh, nên G0-Tier 1 dùng toàn bộ validation thay cho 2 000 điểm.
- **A2–A3.** H3 gần như đúng theo cấu trúc: G0 lọc cặp null theo tỉ lệ đồng ý ≥ 0.98, còn các cặp drift có vùng bất đồng khoảng 10%. Found-rate của Δ-CE là một bộ ước lượng cục bộ của thể tích bất đồng (định nghĩa của Hammer & Hüllermeier). Giá trị của Δ-CE là trả lời *cái gì* và *ở đâu* thay đổi, không phải *bao nhiêu*.
- **A4.** Khi đã trích DIG và DeepXplore, H2 chỉ so với B3 (mù với thay đổi phi cục bộ theo thiết kế) dễ bị coi là so với straw-man.
- **A5.** Thứ tự đóng góp theo độ chắc: benchmark → so sánh có kiểm soát → witness margin-κ.
- **A6.** arXiv v2 đã có phần regularization trùng tiêu đề bản journal.

## Làm rõ khi triển khai (Research Lead, cùng ngày)

- **B0 tính trên đâu.** Điểm đánh giá (spec §7) được lọc sao cho cả hai mô hình cùng dự đoán đúng với margin ≥ κ, nên tỉ lệ bất đồng trên chính các điểm đó luôn bằng 0. Vì vậy B0 được tính trên **toàn bộ tập test của tier, không lọc** (ước lượng Monte Carlo của thể tích bất đồng theo phân phối dữ liệu). Xem D-012.
