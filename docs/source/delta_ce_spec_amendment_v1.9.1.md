# Δ-CE spec — Amendment v1.9.1 (D-037)

**Ngày:** 19/10/2026 · **Người viết:** Research Lead
**Bối cảnh:** viết **sau** pilot qualification exp_0024 (α̂ = 0.233, t̂ = 0.630, độ chính xác 0.929; trượt |t̂ − t| ≤ 0.02). Pilot chỉ đo qualification: chưa chạy phương pháp giải thích nào và chưa có kết quả nào liên quan tới H5a–H5d. Amendment này chỉ sửa tiêu chí kiểm tra thao tác. **Không đổi** giả thuyết, ngưỡng xác nhận, công thức fine-tune, lưới, seed, phương pháp, test set hay cổng quyết định của v1.9.

## 1. Lý do

1. Tiêu chí (2) của §2.3 v1.9 (|t̂ − t| ≤ 0.02) được hiệu chuẩn sai: chính các cặp T4 đã khóa cho t̂ = 0.598 / 0.621 / 0.591, và cặp 7 trượt tiêu chí này.
2. H5a–H5d không đòi hỏi t̂ = t. Theo Proposition 2, chúng chỉ đòi hỏi (i) góc xoay được tạo ra và (ii) các ô được so sánh có offset thực tế khác nhau. Tiêu chí mới kiểm đúng hai điều này.

## 2. Thay §2.3 (2) bằng

Mọi cặp vẫn phải đạt (1) |α̂ − α| ≤ 0.05 + 0.3α và (3) độ chính xác theo luật mới ≥ 0.90. Tiêu chí (2) được thay bằng các kiểm tra offset sau. Chúng tính trên t̂ đã đo, theo từng seed.

- **(2a) Có offset ở hàng α = 0:** với t ∈ {0.53, 0.56, 0.60}, t̂(0, t) − t̂_old ≥ 0.02, trong đó t̂_old là t̂ của f_old cùng seed.
- **(2b) Thứ tự offset:** trong mỗi hàng α, t̂ tăng ngặt theo t. Một cặp vi phạm thứ tự chỉ bị loại khỏi những kiểm định dùng tới ô của nó.
- **(2c) Điều kiện riêng cho H5c:** t̂(0.25, 0.60) − t̂(0.25, 0.50) ≥ 0.05 ở mỗi seed được dùng. Seed nào không đạt thì không dùng cho H5c.

Luật dừng 1/3 áp dụng trên tiêu chí (1), (3) và (2a). Tiêu chí (2b) và (2c) chỉ loại cặp khỏi kiểm định cụ thể.

**Ô null (0, 0.50):** giữ tiêu chí DD ≤ 0.02. Nếu trượt, ô này vẫn được dùng làm mốc t̂ và báo cáo found-rate như mô tả, nhưng không được gọi là null. Ô này không tham gia H5a–H5d.

## 3. Cách gọi tên trong báo cáo

Mọi bảng và hình dùng (α̂, t̂) đã đo. Các ô t = 0.5 và α > 0 (dùng cho H5d) được gọi là "rotation with measured offset t̂ − t̂_old", không gọi là "pure rotation", nếu t̂ − t̂_old > 0.02. Diễn giải định trước của H5d trong v1.9 §3 giữ nguyên. Riêng câu "T1 không phải phép xoay sạch" chỉ được dùng nếu H5d đạt.

## 4. Thực thi

- **Chỉ chạy trên CPU với solver đã khóa** cho mọi bước của Stage A, không dùng GPU, để phép kiểm nhất quán §5.3 so sánh được với T4.
- **Đường cong κ thu hẹp:** κ ∈ {0, 0.25, 1.0} chỉ chạy trên hàng α = 0 và ô null. Mọi ô khác chỉ chạy κ = 0.5.
- **Ngân sách:** Stage A ≤ 30 CPU-giờ. Vượt thì dừng và báo.

## 5. Xác nhận các cách hiểu của triển khai (exp_0024)

1. sign(m_n) là quyết định của f_new, tức dấu của logit₁ − logit₀.
2. Hồi quy logistic không phạt (MLE). Ridge chỉ dùng làm phép kiểm độ nhạy.
3. pair_id = 20…51, thứ tự α → t → seed, seed = base trong tier0.yaml + pair_id, test seed = 7000 + pair_id.
4. Công thức fine-tune: 10 epoch, lr 10⁻³, batch 256, không cosine (knob T4 của exp_0003).
