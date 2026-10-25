# Lịch sử kết quả thực nghiệm Δ-CE

Tài liệu này gom kết quả các thực nghiệm từ khởi tạo benchmark đến exp_0040. Đây là bản chỉ mục và tổng thuật; số liệu chi tiết, cấu hình, log và artifact gốc vẫn nằm trong `runs/experiments/`. Kết quả xác nhận được phân biệt với kiểm tra kỹ thuật, phân tích mô tả và thí nghiệm bị dừng. Chỉ số Recovery@r là tỷ lệ điểm đánh giá mà phương pháp vừa tìm được lời giải trong ngân sách khoảng cách, vừa xếp yếu tố bị can thiệp là yếu tố thay đổi.

## Kết luận chính

- Mục tiêu ban đầu Route A **không đạt**: trên Tier 0, Δ-CE không vượt B3 ở T3; chênh lệch Recovery@r là −0.091 (95% CI [−0.133, −0.050]). Tier 1 bị loại ở cổng validation trước khi chạy kiểm định cuối.
- Kết quả so sánh phụ thuộc dạng thay đổi. Trên T4, Δ-CE vượt B3 +0.312 (95% CI [+0.273, +0.352]) ở bộ xác nhận mới; trên T1, B3 vượt Δ-CE. Δ-CE không tìm witness trong 8 cặp null Tier 0 đã kiểm tra.
- Các thí nghiệm mở rộng cho thấy ngưỡng κ có ý nghĩa quyết định: dưới ngưỡng logit 2κ, Δ-CE từ chối đưa ra witness dù CE differencing vẫn đổi; trên ngưỡng, witness xuất hiện theo dự đoán trong thiết kế offset chính xác.
- Phân tích tại vị trí witness giải thích một phần thất bại của phép chiếu tại điểm gốc: chênh lệch Recovery chiếu đổi từ −0.190 tại (z0,z0) sang +0.029 tại (z*,z0), kiểm định H8 đạt p_Holm = 0.0006. Kết quả này không đảo ngược kết luận Route A.
- Kiểm tra độ ổn định trên validation cho thấy kết quả Shortcut Δ-CE thấp hơn CF-diff trong cả 5 lần chạy (khoảng cách 0.173–0.183). H8 cũng giữ quy tắc làm hẹp trong cả 5 lần chạy (T = 0.085–0.120).

## Diễn tiến theo giai đoạn

| Giai đoạn | Run | Kết quả và ý nghĩa |
|---|---|---|
| Dựng benchmark và qualification Tier 0 | exp_0001–0004 | exp_0001–0003 cho thấy quy trình ban đầu không tạo đủ drift hợp lệ; các điều chỉnh được ghi trước lần đánh giá qualification tiếp theo. exp_0004 đạt tiêu chí G0 trên 17/17 cặp Tier 0. Đây là xác nhận benchmark, chưa phải so sánh phương pháp. |
| Kiểm solver tuyến tính | exp_0005–0007 | exp_0005 không đạt H4; exp_0006 cải thiện độ chính xác nhưng tỷ lệ tìm được nghiệm Δ-CE ở p=5 chỉ 0.967. Sau sửa solver và khóa quy tắc, exp_0007 đạt H4 trên hai bộ seed: 100% tỷ lệ share theo tiêu chí cho Δ-CE và CE, sai số tương đối lớn nhất 5.8×10⁻⁴. |
| Pilot, hiệu năng, calibration | exp_0008–0011 | r Tier 0 được khóa ở 0.5413. Pilot ban đầu đạt found-rate nhưng không đạt dự báo thời gian do tối ưu hóa trên điểm không có witness; bước tối ưu được tăng tốc bit-identical 1.7–2.2×. exp_0011 cho thấy found-rate không đổi và thời lượng dự báo Tier 0 16.5 giờ, đạt cổng G1. |
| Tier 1 / MorphoMNIST | exp_0012–0016 | Dữ liệu và mô hình được chuẩn bị, nhưng các vòng G2 không đạt điều kiện về shape oracle/f_old và drift T1/T3. Ở vòng cuối, f_old đạt 0.984–0.987 và N0 đạt, nhưng T1/T3 vẫn trượt mọi biến thể. Theo quy tắc định trước, Tier 1 bị loại; H1/H2b không được kiểm định trên test. |
| Đánh giá Tier 0 chính | exp_0017 | 17 cặp × 300 điểm trên test. T1: Recovery Δ-CE 0.376, B1 0.542, B3 0.909. T3: Δ-CE 0.529, B1 0.316, B3 0.620; so sánh xác nhận Δ-CE − B3 = −0.091 (95% CI [−0.133, −0.050]), nên H2a bị bác bỏ. T4: Δ-CE 0.466, B3 0.070, B1 0.766. Null N0/N1: Δ-CE found-rate = 0; B1/B3 thường vẫn trả attribution. H3 AUROC bằng 1.0 cho Δ-CE, B0, B1, B3 trên các cặp đã qua G0, do đó không phân biệt được phương pháp. |
| Tái lập và xác nhận mới | exp_0018–0022 | exp_0018 tái chạy cặp drift T3 và null N0 từ checkout sạch, bit-identical với exp_0017. exp_0019 trên test mới xác nhận T4: Δ-CE − B3 = +0.312 (95% CI [+0.273, +0.352]); null: Δ-CE không có witness trên cả 8 cặp. T1 vẫn bất lợi: chênh lệch Recovery chiếu Δ-CE − B3 = −0.304, nên C3 không đạt. exp_0020 Tier 1 N0 là phân tích thăm dò: Δ-CE không có witness ở 4/5 cặp, một cặp 47.5%; B1 khoảng 0.98. exp_0021 cho T5 không đạt G0 ở mọi seed/recipe (DD ≤ 0.0005 < 0.02), nên exp_0022 và kiểm định test C4–C6 không chạy. |
| Rotation × offset và dữ liệu bảng Adult | exp_0023–0029 | Kiểm tra số học P5 trên Tier L đạt điều kiện khoảng cách nhưng tiêu chí hướng/tie thất bại ở p=5 do hướng tiếp tuyến phẳng; có một lần solver bỏ lỡ phụ thuộc batch ở trường hợp offset song song. Trên grid CNN, H5a được ủng hộ: B3 Recovery 0.081 [0.066, 0.096], H5b được ủng hộ: tăng theo góc +0.570 [0.527, 0.613]; H5c không được ủng hộ (−0.098, CI90 [−0.143, −0.053]); H5d không được ủng hộ (Δ-CE − B3 sau chiếu −0.352, CI90 [−0.378, −0.327]). Trên 5 cặp Adult null, Δ-CE N∪O = 0 ở mọi cặp, B1-det 0.977–1.000 (H6 đạt); diễn giải giới hạn ở abstention vì không có decoder manifold và decisive disagreement hiếm. Chẩn đoán offset nhỏ cho thấy không thể dùng offset đơn lẻ để phân loại có/không witness ở vùng 0.03–0.04. |
| Logit offset, H8, abstention | exp_0030–0035 | exp_0030 validation gợi ý phép chiếu tại witness làm hẹp khoảng cách; exp_0031–0035 hoàn tất các nhánh H7c/H8 và kiểm tra abstention/κ. H7c đạt trên test: B1-det found-rate 0.961 [0.955, 0.967], từng cặp 0.927–0.997; dưới ngưỡng 2κ Δ-CE không tìm witness, còn trên ngưỡng witness chỉ xuất hiện ở lớp bị hạ margin. H8 đạt: gap −0.190 [−0.222, −0.159] ở (z0,z0) chuyển thành +0.029 [0.006, 0.053] ở (z*,z0); T=0.132, CI95 lower 0.065, p_Holm=0.0006. T-C không có false negative trong 400 truy vấn null và 150 truy vấn lớp bất khả thi; G3 có 3/395 false negative, tất cả ở một cặp biên mỏng; sai khác máy được công bố. T-D cho found-rate theo κ đúng với Lemma 1: ngưỡng tại |b|=2κ và chỉ một lớp; nhiễu do retraining giảm khi κ tăng. |
| T1 2×2 và kiểm tra solver | exp_0036–0039 | exp_0036 khảo sát H8 trên T1: gap đổi −0.166 → −0.024 khi lấy pháp tuyến tại witness, mô tả chứ không phải kiểm định. exp_0037 cho thấy nghiệm tuyến tính KKT/SLSQP độc lập đạt 100%, nhưng tiêu chí vector định trước thất bại ở p=5; sai khác của solver thực nghiệm tập trung ở thành phần tiếp tuyến. exp_0038: khi giải lại cùng điểm, top-factor thay đổi 0.5% ở Δ-CE (2/409), 10.9% ở CF-diff (61/561), 33% ở DeepXplore (98/301); p95 sai số vector Δ-CE có đuôi nặng (0.22). exp_0039: Recovery ổn định giữa 5 lần chạy; SD Δ-CE ≤ 0.005, DeepXplore 0.013–0.028; dấu chênh lệch Δ-CE − DeepXplore không đổi ở Subgroup và Threshold. |
| Độ biến thiên lần chạy cuối | exp_0040 | Trên validation, Shortcut Δ-CE Recovery 0.320–0.330 so với CF-diff 0.503; chênh lệch −0.173 đến −0.183 trong cả 5 lần chạy. Threshold: Δ-CE 0.453–0.457, CF-diff 0.757–0.783; chênh lệch −0.300 đến −0.330. H8 có T=0.085–0.120 > 0 trong cả 5 lần chạy. Kết luận đã khóa: bài ESANN không đổi; phân tích bổ sung ở `research/theory/proofs.md` §6. |

## Các kết quả chính trên Tier 0

| Can thiệp | Δ-CE Recovery@r | B1 Recovery@r | B3 Recovery@r | Đọc kết quả |
|---|---:|---:|---:|---|
| T1, exp_0017 | 0.376 | 0.542 | 0.909 | B3 tốt hơn; tie tuyến tính dự đoán không được xác nhận. |
| T3, exp_0017 | 0.529 | 0.316 | 0.620 | H2a bị bác bỏ; B3 tốt hơn Δ-CE có ý nghĩa. |
| T4, exp_0017 | 0.466 | 0.766 | 0.070 | Δ-CE hơn B3 mô tả rõ, nhưng kém B1. |
| T4, test mới exp_0019 | 0.408 | — | 0.096 | C1 xác nhận Δ-CE hơn B3 +0.312, nhưng không chứng minh hơn B1. |
| Null N0/N1, exp_0017 | found 0.000 | found ≈0.995 | found 1.000 | Δ-CE abstains trên các null đã qualify; các baseline attribution vẫn trả lời. |

Điểm T4 ở exp_0017 là trung bình mô tả 3 cặp; C1 ở exp_0019 là kiểm định xác nhận đã đăng ký trên test mới. Không gộp hai con số thành một ước lượng duy nhất.

## Trạng thái mục tiêu và giới hạn

Theo quy tắc đánh giá đã định, kết quả của dự án là Route B (benchmark + analysis), không phải thành công Route A. Tier 1 đã bị dừng đúng quy tắc trước kiểm định test; T5 bị dừng ở G0; các phân tích v1.9/v1.10 mở rộng kết luận cơ chế nhưng không thay thế mục tiêu chính. Kết quả Tier 0 chính có tái lập độc lập cho hai cặp ở exp_0018. Kiểm tra tại exp_0038–0040 dùng validation, mang tính chẩn đoán/ổn định, không phải phép xác nhận mới trên test.

## Nguồn số liệu gốc

- Tier 0 và các kiểm định bổ sung: `research/results_summary.md`, `runs/experiments/exp_0017/results.json`, `runs/experiments/exp_0019/`.
- v1.9 rotation × offset và Adult: `research/results_v19.md`, `runs/experiments/exp_0026/`, `runs/experiments/exp_0028/`.
- H7c/H8/T-C/T-D và H8 T1: `research/HYPOTHESES.md` H-010/H-011, `research/DECISIONS.md` D-043–D-047, `runs/experiments/exp_0032/`–`exp_0036/`.
- Độ ổn định/độ nhạy solver: `runs/experiments/exp_0037/`–`exp_0040/`, `research/theory/proofs.md` §6.
- Bài hiện tại: `paper/main.pdf`; nguồn LaTeX: `paper/main.tex`.
