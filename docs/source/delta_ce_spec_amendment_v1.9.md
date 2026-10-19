# Δ-CE spec — Amendment v1.9 (D-036)

**Ngày:** 19/10/2026 · **Người viết:** Research Lead · **Trạng thái:** khai báo trước khi chạy, phải được commit **trước** khi sinh bất kỳ cặp mô hình hay test set nào của amendment này.

**Không thay đổi:** mọi kết quả đã khóa (exp_0017, exp_0018, exp_0019, D-034, D-035), commit đã khóa ede7374, κ = 0.5, r = 0.54127, solver hai pha, định nghĩa Recovery@r, found-rate, Rec_proj (như C3), bootstrap 10 000 lần phân tầng theo cặp. Bản paper hiện tại (paper/main.tex, 6 trang) là **phương án dự phòng** và không bị sửa cho tới cổng quyết định (§7).

---

## 1. Mục đích

Bản hiện tại chỉ có một mức cường độ cho mỗi kiểu thay đổi. Amendment này kiểm tra xem các luật phát hiện suy ra từ Proposition 2 (tức P5: rotation và offset) có đúng trên CNN khi thay đổi được điều khiển **liên tục**, và kiểm lại C3 trên một phép xoay sạch. Stage B (tùy chọn) kiểm lại kết quả abstention (C2) trên dữ liệu thật.

## 2. Stage A — lưới rotation × offset trên Tier 0

### 2.1 Thay đổi được cấy

Dữ liệu thích nghi được gán nhãn theo luật

  y = 𝟙[ aspect + α·(colour − 0.5) > t ]

- α ∈ {0, 0.1, 0.25, 0.5} (góc xoay danh nghĩa θ = arctan α ≈ 0°, 5.7°, 14.0°, 26.6°)
- t ∈ {0.50, 0.53, 0.56, 0.60}
- Ô (α=0, t=0.50) là cặp null kiểu N0 (luật không đổi) và được dùng làm đối chứng.
- Ô (α=0, t=0.60) lặp lại thiết kế T4 và được dùng làm phép kiểm nhất quán (§5.3).

**16 ô × 2 seed f_old (seed 0 và 1) = 32 cặp.** Quy trình fine-tune giống T4: 10 000 mẫu, nhiễu nhãn 5%, 10 epoch, lr 10⁻³. Dữ liệu thích nghi lấy mẫu factor đều trên [0,1]⁵ như dữ liệu train gốc.

### 2.2 Factor đích k*

- α = 0, t > 0.5 (offset thuần): k* = aspect.
- α > 0: k* = colour (phụ thuộc mới). Recovery theo aspect của các ô α > 0 chỉ được báo cáo mô tả.
- Ô (0, 0.50): không có k*, chỉ đo found-rate.

### 2.3 Đủ điều kiện (qualification), đo trên validation

Với mỗi cặp, fit hồi quy logistic của sign(m_n) trên 5 factor, dùng 5 000 điểm validation, rồi suy ra α̂ = w_colour / w_aspect và t̂ (điểm biên khi colour = 0.5, các factor khác tại 0.5). Cặp đủ điều kiện khi thỏa cả ba:

1. |α̂ − α| ≤ 0.05 + 0.3α;
2. |t̂ − t| ≤ 0.02;
3. độ chính xác của f_new theo luật mới trên validation ≥ 0.90.

Ô (0, 0.50) dùng tiêu chí null hiện có (bất đồng quyết đoán ≤ 0.02).

**Luật dừng:** nếu quá 1/3 số cặp không đủ điều kiện, Stage A chuyển sang mô tả hoàn toàn (không kiểm định H5) và được báo cáo như vậy. Các cặp không đủ điều kiện bị loại khỏi kiểm định và được liệt kê.

Mọi kiểm định và hình vẽ đều báo cáo kèm (α̂, t̂) đo được, không chỉ giá trị danh nghĩa.

### 2.4 Phương pháp

Δ-CE (κ = 0.5), B1-det, B3, A2. Bỏ B1, B4, B5, A1 để tiết kiệm (B1 ≈ B1-det trong mọi điều kiện đã đo). Thêm phân tích mô tả: found-rate của Δ-CE với κ ∈ {0, 0.25, 0.5, 1.0} trên mọi ô (đường cong abstention).

### 2.5 Dữ liệu test

Test set mới từ renderer: 300 điểm hợp lệ mỗi cặp, seed 7000 + chỉ số cặp. Validation chỉ dùng cho qualification và pilot (§6). Không xem kết quả của Δ-CE, B1-det, B3 hay A2 trên validation trước khi chạy test.

## 3. Giả thuyết khai báo trước (Stage A)

Mỗi hiệu số dùng bootstrap ghép cặp, phân tầng theo cặp, 10 000 lần, seed 6500. Hiệu chỉnh Holm cho H5a–H5d (m = 4). Nếu một ô liên quan có cặp bị loại, kiểm định dùng các cặp còn lại. Nếu ô đó không còn cặp nào, giả thuyết được báo cáo là "không kiểm được".

| ID | Dự đoán (từ Prop. 2) | Kiểm định | Xác nhận khi |
|---|---|---|---|
| **H5a** | Offset làm B3 mù | Recovery(B3) gộp các ô α = 0, t ∈ {0.53, 0.56, 0.60} | cận trên CI95 ≤ 0.20 |
| **H5b** | B3 thấy rotation và tăng theo góc | Rec(B3; α=0.5, t=0.5) − Rec(B3; α=0.1, t=0.5) | cận dưới CI95 > 0 (p_Holm < 0.05) |
| **H5c** | B3 không phụ thuộc offset | Rec(B3; α=0.25, t=0.60) − Rec(B3; α=0.25, t=0.50) | CI90 nằm trong [−0.10, 0.10] (TOST) |
| **H5d** | Trên phép xoay sạch, Δ-CE và B3 trùng nhau sau khi chiếu (kiểm lại C3) | Rec_proj(Δ-CE) − Rec_proj(B3), gộp các ô α ∈ {0.25, 0.5}, t = 0.5 | CI90 nằm trong [−0.10, 0.10] (TOST) |

**Diễn giải định trước cho H5d:**
- Nếu đạt, C3 trượt trên T1 được quy cho việc T1 không phải một phép xoay sạch (shortcut học từ tương quan), và câu "T1 unexplained" được thay bằng điều này.
- Nếu trượt, paper ghi rằng phân tích tuyến tính không chuyển sang CNN ngay cả với phép xoay sạch.

**Dự đoán mô tả (không kiểm định):**
- (i) Found-rate của Δ-CE ở α = 0 tăng theo t.
- (ii) Ở α = 0, witness chỉ xuất hiện ở lớp y = 1, như T4.
- (iii) A2 có Recovery cao ở các ô α = 0, như T4. Đây là lời nhắc rằng Recovery đo độ mù, không đo độ nhạy.

## 4. Stage B — abstention trên dữ liệu bảng thật (có điều kiện)

**Điều kiện bắt đầu:** Stage A đã chạy xong toàn bộ test trước 23:59 ngày 31/10. Nếu không, Stage B bị hủy và ghi vào DECISIONS.

- **Dữ liệu:** Adult (OpenML id 1590), chỉ dùng 6 đặc trưng số (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week), chuẩn hóa theo train. Không gian tìm kiếm là không gian đặc trưng đã chuẩn hóa, D là đồng nhất, ràng buộc hộp theo phân vị 1–99% của train. Chia train/val/test là 60/20/20 với seed 2026. Nếu không tải được Adult, dừng và báo; không tự chọn bộ khác.
- **Mô hình:** f_old là MLP 2 lớp ẩn 64 đơn vị, 5 seed. f_new là f_old fine-tune trên một mẫu mới của train (nhãn thật, thứ tự mới), 3 epoch, theo kiểu N0. **5 cặp null.**
- **Ngân sách r:** áp dụng một lần quy tắc hiệu chuẩn r của spec trên validation Stage B, rồi đóng băng trước test. κ = 0.5.
- **Đủ điều kiện:** bất đồng quyết đoán trên validation ≤ 0.02 và độ chính xác của f_old ≥ 0.80.
- **Giả thuyết H6 (khai báo trước, kiểm định riêng, không vào họ Holm của H5):** trên test, 300 điểm hợp lệ mỗi cặp, found-rate của Δ-CE ≤ 0.05 trên **mọi** cặp đủ điều kiện, và found-rate của B1-det ≥ 0.90 trên mọi cặp đủ điều kiện. Xác nhận khi cả hai vế đúng.
- **Giới hạn ghi trước:** dữ liệu bảng không có decoder lên đa tạp, nên witness có thể không thực tế. Stage B chỉ kiểm abstention, không kiểm attribution.

## 5. Báo cáo và kiểm tra

1. Kết quả H5a–H5d và H6 được báo cáo đúng như bảng trên, kể cả khi trượt.
2. Hình bản đồ: lưới α̂ × t̂, ba panel (Recovery của B3, Recovery của Δ-CE, found-rate của Δ-CE). Hình này thay Hình 1 của paper nếu Stage A được dùng.
3. **Kiểm nhất quán:** ô (0, 0.60) phải cho Recovery(B3) ≤ 0.20 và found-rate của Δ-CE trong [0.30, 0.65], tức gần T4 đã khóa (0.07 và 0.47). Nếu lệch, dừng phân tích và báo trước khi diễn giải bất cứ điều gì.
4. Mọi kết quả nằm trong research/results_v19.md, tách khỏi results_summary.md.

## 6. Pilot

Chỉ một ô (α = 0.25, t = 0.56), seed 0, **chỉ trên validation và chỉ cho qualification** (α̂, t̂, độ chính xác). Không chạy phương pháp giải thích nào trong pilot. Nếu pilot không đạt qualification, báo trước khi chạy lưới. Không được tự chỉnh lr, số epoch hay cỡ mẫu.

## 7. Lịch và cổng quyết định

| Hạn | Mốc |
|---|---|
| 21/10 | Commit amendment, cài luật nhãn mới, pilot |
| 26/10 | Toàn bộ 32 cặp đã fine-tune và qualification |
| 31/10 | Stage A test xong (điều kiện cho Stage B) |
| 5/11 | Stage B xong (nếu chạy) |
| **9/11** | **Cổng quyết định:** Research Lead chọn bản nộp |

Tại cổng, nếu Stage A chưa có đủ kết quả test thì nộp bản 6 trang hiện tại, không thêm gì từ v1.9. Nếu có, paper được viết lại để thêm bản đồ và H5. Stage B (nếu có) thêm một câu vào phần abstention và phần Limitations (1).

**Ngân sách tính toán:** Stage A ≤ 4 GPU-giờ, Stage B ≤ 1 GPU-giờ. Vượt thì dừng và báo.

## 8. Ghi chú về nguồn gốc

Proposition 2 trong paper chính là P5 (research/theory/P5_rotation_offset.md), được viết ngày 18/10. Các dự đoán H5a–H5d được viết hôm nay, 19/10, trước khi có bất kỳ cặp mô hình nào của lưới này. Commit của file này là bằng chứng thời điểm.
