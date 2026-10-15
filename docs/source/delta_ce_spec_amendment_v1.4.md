# Spec amendment v1.4 — Δ-CE (sau H4, exp_0005)

**Quyết định của tác giả, 15/10/2026.** Trước mọi kết quả của phương pháp trên dữ liệu (Tier L là tổng hợp, không có split test). Không nới dung sai H4. Quyết định tương ứng: `research/DECISIONS.md` D-022.

```text
Spec amendment v1.4 — sau H4 (exp_0005), 15/10/2026. Trước mọi kết quả của phương pháp trên dữ liệu
(Tier L là tổng hợp, không có split test). Không nới dung sai H4.

E1. Bộ giải (Δ-CE loại N/O và CE thường) gồm hai pha:
    Pha 1: spec §6.1 + R1 (λ giảm một nửa mỗi T/8 chừng nào chưa tìm thấy điểm khả thi), R = 5, T = 400.
    Pha 2: với ứng viên khả thi tốt nhất của MỖI restart, chạy SLSQP (scipy.optimize, gradient từ torch):
        min ||z − z0||² (+ μ·Reg(z) ở Tier 1, như mục tiêu gốc)
        s.t. ràng buộc witness (Δ-CE) hoặc m(z) ≤ −κ (CE thường); Tier 0: bounds [0,1]^5.
        maxiter = 100, ftol = 1e-10.
    Kết quả: nghiệm khả thi (kiểm lại bằng forward pass) có d nhỏ nhất qua các restart;
    SLSQP thất bại hoặc ra điểm không khả thi → giữ ứng viên pha 1 của restart đó.
    Không có ứng viên khả thi → NOT_FOUND. Ngân sách r áp dụng sau cùng như cũ.
E2. Áp dụng đồng đều cho Δ-CE, CE trong B1, B1-det, A2, A3 và hiệu chuẩn r.
    B3, B5 không đổi. A1 (λ = 0) chỉ dùng pha 1.
E3. H4 mở rộng (khai báo trước): ngoài Δ-CE, kiểm thêm CE thường so với nghiệm đóng
    của CE tuyến tính (u = −(m(z0)+κ)/‖w‖² · w), cùng dung sai.
    Đây là kiểm tra fidelity của B1 mà D-007 đã yêu cầu.
E4. Quy tắc dừng: chạy H4 MỘT lần với bộ giải v1.4.
    Đạt → tiếp tục G1. Trượt → dừng và báo cáo; không thử biến thể khác.
    Ngoại lệ duy nhất là sửa bug có ghi lại (code không giải đúng bài toán đã khai báo), không phải đổi thuật toán.
E5. Runtime đo ở pilot G1 với bộ giải mới; tiêu chí ≤ 5 phút mỗi cặp giữ nguyên.
```

## Lý do (tóm tắt từ tác giả)

- Adam với lr cố định trên bài toán phạt hinge về bản chất không hội tụ tới nghiệm KKT; nó dao động quanh nghiệm, lịch λ không thu nhỏ được dao động. p=5 khó vì w_o, w_n gần song song (miền khả thi là một nêm hẹp, ill-conditioned) — tình huống này cũng xảy ra với cập nhật mô hình thật và Tier 0 có 5 chiều.
- Bộ giải phạt + Adam trong spec là lựa chọn tùy ý. Bài toán là cực tiểu khoảng cách có ràng buộc trong không gian ít chiều (5 / 16), nên giải bằng bộ giải có ràng buộc thực sự. Với bài toán tuyến tính SLSQP về đúng KKT ở độ chính xác máy.
- Công bằng: CE thường dùng cùng hai pha; nếu chỉ Δ-CE được polish, B1 thua vì nhiễu tối ưu chứ không phải vì phương pháp. B3 có dạng đóng, B5 là line search, A1 không có mục tiêu khoảng cách.
- (B) nới dung sai bị loại: dời cột gôn của một gate kiểm tra tính đúng.
