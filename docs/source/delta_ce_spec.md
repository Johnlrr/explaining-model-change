# SPEC — Differential Counterfactuals (Δ-CE): minimal witnesses of changing decision behaviour

**Phiên bản:** spec v1.0 (bản nháp để khóa) — 15/10/2026
**Khóa spec (freeze):** 17/10/2026, 23:59 ICT. Sau mốc này, mọi thay đổi phải ghi vào `logs/decisions.md` (§13).
**Tài liệu gốc:** `delta_ce_proposal.md` v1. Spec này **thay thế** proposal ở mọi điểm khác nhau (§1).
**Venue:** ESANN 2027, special session *From Understanding Distributional Change to Informed Action*. Nộp nội bộ 16/11, hạn cứng 18/11/2026.

---

## 1. Thay đổi so với proposal v1 (sau khi đọc full text và trang ESANN)

| # | Thay đổi | Lý do |
|---|---|---|
| C1 | Nhắm special session *From Understanding Distributional Change to Informed Action* (Kuhl, Vaquet, Hinder và cộng sự). Đóng khung bài toán là **thay đổi hành vi quyết định sau drift và model adaptation** | Chủ đề của session có "Explainable AI for … concept drift, and changing decision behavior", "Definitions of meaningful change", "Drift characterization, attribution and localization", "Realistic drift benchmarks". Ban tổ chức thuộc nhóm Bielefeld, cũng là nhóm của Artelt et al. |
| C2 | Thêm **lập luận định vị bắt buộc** đối với Artelt et al. §3 (xem §2) | Artelt et al. đã xét tập bất đồng $\{x: h(x)\neq h'(x)\}$ rồi gạt sang bên vì nó "có thể nhỏ và mô tả không có ý nghĩa" |
| C3 | Baseline B1 = đúng Định nghĩa 3 của Artelt (so sánh hai CE theo từng thành phần). Baseline B3 = đúng xấp xỉ gradient (Eq. 14) của Artelt | Tái hiện trung thành phương pháp gốc, tránh bị chê là straw-man |
| C4 | Thêm điều kiện **N1 = virtual drift** (phân phối input đổi, quy tắc gán nhãn giữ nguyên) | Kiểm tra trực tiếp khái niệm "meaningful change" của session |
| C5 | Đổi H3 từ SAR sang **phát hiện thay đổi ở mức cặp mô hình (AUROC)**. SAR chuyển thành metric phụ | SAR phụ thuộc ngưỡng độ lớn, khó so sánh công bằng giữa các phương pháp |
| C6 | Nêu rõ giới hạn **hành vi vs cơ chế** | Δ-CE chỉ thấy thay đổi *hành vi* trên manifold. Artelt et al. nhắm thay đổi *quy tắc* ngay cả khi dự đoán trùng nhau |
| C7 | Ngân sách trang: 6 trang **gồm** hình và references, không appendix. Chứng minh đầy đủ đặt ở link GitHub (không được review) | Theo author guidelines |
| C8 | Oracle Tier 1 dùng **nghiệm đóng theo từng pixel** (kênh G không mang màu đối tượng) | Không cần ngưỡng hay auditor học từ dữ liệu |

---

## 2. Định vị so với Artelt et al. (phải xuất hiện trong paper)

Artelt et al. (arXiv 2104.02459v2) có các nội dung sau:
- §3: nêu cách "hiển nhiên" là giải thích vùng $\{x\mid h(x)\neq h'(x)\}$, nhưng bỏ qua vì với adaptation tăng dần thì tập này có thể nhỏ. Thay vào đó họ so sánh CE của hai mô hình như một proxy cho *quy tắc khái quát hóa*.
- Định nghĩa 3: so sánh hai CE bằng $\psi$ = trị tuyệt đối hiệu từng thành phần, và $\Psi$ = khoảng cách Euclid hoặc cosine.
- Định lý 1–2: chỉ cho bộ phân lớp tuyến tính $\mathrm{sign}(w^\top x)$, $\|w\|=1$, **không có bias**.
- §4.2: interest function dùng xấp xỉ $\mathrm{CF}(x,h)\approx x-\eta h(x)\nabla f(x)$.
- Thực nghiệm chỉ trên tabular, hyperspectral và HAR, với GNB, hồi quy tuyến tính, logistic, decision tree. **Không có ground truth định lượng**; đánh giá bằng quan sát hình.

**Bốn luận điểm định vị:**

1. **Witness không cần dữ liệu bất đồng.** Δ-CE không mô tả các *mẫu quan sát được* có dự đoán khác nhau. Nó *tìm* witness gần nhất trên manifold, kể cả khi không có mẫu test nào nằm trong vùng bất đồng. Bổ đề 1 mô tả chính xác khi nào vùng bất đồng (với margin $\kappa$) rỗng, nên lời phê bình "tập có thể nhỏ" trở thành một **chứng chỉ null** thay vì một điểm yếu.
2. **Tính nhất quán với Định lý 1 của Artelt.** Mệnh đề 4 của chúng tôi cho thấy cả Δ-CE lẫn hiệu CE đều khôi phục $P\Delta w$ trong trường hợp tuyến tính. Điều này khớp với Định lý 1 của họ. Đóng góp không nằm ở hướng, mà ở (i) trường hợp có bias và margin, (ii) chứng chỉ null, (iii) thay đổi phi cục bộ, và (iv) benchmark định lượng có ground truth.
3. **Thay đổi phi cục bộ.** Xấp xỉ gradient (Eq. 14 của họ) chỉ nhìn thấy thay đổi tại $x$. Điều kiện T3 dựng đúng tình huống mà xấp xỉ này mù (§3.6 của proposal).
4. **Giới hạn trung thực.** Δ-CE nhận diện thay đổi **hành vi** trên manifold trong ngân sách $r$. Nếu hai mô hình đổi quy tắc nội bộ nhưng không đổi quyết định ở đâu trên manifold, Δ-CE trả về "không thay đổi", còn cách của Artelt có thể vẫn báo khác biệt. Chúng tôi trình bày điều này như một **định nghĩa có chủ đích của "meaningful change"**, và kiểm bằng điều kiện N1.

---

## 3. Định nghĩa hình thức (đóng băng)

### 3.1. Ký hiệu

- $\mathcal Z$: không gian tìm kiếm với metric Euclid $d$. Tier 0 dùng $[0,1]^5$; Tier 1 dùng $\mathbb R^{16}$ (latent VAE).
- $D:\mathcal Z\to$ ảnh: renderer (Tier 0) hoặc decoder (Tier 1).
- Classifier nhị phân với logit $f(\cdot)\in\mathbb R^2$. Margin của lớp $y$: $m(z)=f_y(D(z))-f_{1-y}(D(z))$.
- $m_o, m_n$ là margin của $f_{old}, f_{new}$; $h=m_n-m_o$.
- $\kappa=0.5$ (logit-margin, tương ứng xác suất khoảng 0.62).

### 3.2. Δ-CE

**Điểm hợp lệ:** $z_0$ thỏa $m_o(z_0)\ge\kappa$ và $m_n(z_0)\ge\kappa$ (cả hai dự đoán $y$ một cách dứt khoát), và $y$ trùng nhãn đúng.

**Loại N** (chỉ $f_{new}$ đổi quyết định):
$$\mathcal F^N_\kappa=\{z:\ m_n(z)\le-\kappa,\ m_o(z)\ge\kappa\}$$

**Loại O** (chỉ $f_{old}$ đổi quyết định):
$$\mathcal F^O_\kappa=\{z:\ m_o(z)\le-\kappa,\ m_n(z)\ge\kappa\}$$

**Δ-CE** là $z^\*\in\arg\min_{z\in\mathcal F}d(z_0,z)$, được trả về khi và chỉ khi $d(z_0,z^\*)\le r$; ngược lại trả về `NOT_FOUND`.

**Quy ước báo cáo:** thí nghiệm chính dùng loại N. Loại O chỉ báo cáo cho N0 và N1 (detection).

### 3.3. Các kết quả lý thuyết sẽ đưa vào paper

Chứng minh đầy đủ ở proposal §3. Trong paper: chứng minh một dòng cho L1 và P2; phác thảo cho P3 và P4; bản đầy đủ đặt ở link.

| ID | Phát biểu | Trạng thái |
|---|---|---|
| L1 | $z\in\mathcal F^N_\kappa\Rightarrow h(z)\le-2\kappa$ và $\lvert h(z)\rvert\ge m_o(z)+\kappa$ | Đã chứng minh |
| P2 | $h$ là $L_h$-Lipschitz trên $B(z_0,r)$ và $h(z_0)+2\kappa>0$ $\Rightarrow$ không có witness trong $B\big(z_0,\min\{r,(h(z_0)+2\kappa)/L_h\}\big)$ | Đã chứng minh; **không** claim là certified cho deep net |
| P3 | Trường hợp tuyến tính có bias: nghiệm đóng hai trường hợp (A: trùng CE thường của $f_{new}$; B: cả hai ràng buộc active) | Đã chứng minh |
| P4 | Trường hợp tuyến tính: $Pu^\*\in\mathrm{span}\{P\Delta w\}$; **hiệu CE cũng vậy** | Đã chứng minh, kèm cảnh báo |

---

## 4. Dữ liệu

### 4.1. Tier L — sanity tuyến tính (H4)

- $p\in\{5,20\}$, 1000 cặp ngẫu nhiên.
- $w_o\sim\mathcal N(0,I)$, $\Delta w\sim\mathcal N(0,0.3^2I)$, $w_n=w_o+\Delta w$; $b_o, b_n\sim\mathcal N(0,0.1^2)$.
- $z_0$ được lấy mẫu cho tới khi $m_o(z_0)\ge\kappa$ và $m_n(z_0)\ge\kappa$.
- So nghiệm của bộ giải (§6) với nghiệm đóng của P3.

### 4.2. Tier 0 — Ellipses (renderer khả vi, manifold chính xác)

**Ảnh:** 32×32×3, tọa độ $(u,v)\in[-1,1]^2$.

**Factor** $\phi\in[0,1]^5$, lấy mẫu độc lập và đều:

| k | Factor | Ánh xạ |
|---|---|---|
| 1 | aspect (factor nhiệm vụ) | $\ell=1.4\,(\phi_1-0.5)$ (log tỉ lệ khung, $\in[-0.7,0.7]$) |
| 2 | size | $s=0.25+0.35\,\phi_2$ |
| 3 | color | màu đối tượng $c=(1-\phi_3)(1,0,0)+\phi_3(0,0,1)$ |
| 4 | bg | độ sáng nền $b=0.1+0.5\,\phi_4$ (xám) |
| 5 | posx | $c_x=0.6\,(\phi_5-0.5)$, $c_y=0$ |

**Hình học:** $A=s\,e^{\ell/2}$, $B=s\,e^{-\ell/2}$.

**Render:**
$$\alpha(u,v)=\sigma\!\Big(20\big(1-\tfrac{(u-c_x)^2}{A^2}-\tfrac{v^2}{B^2}\big)\Big),\qquad I=\alpha\,c+(1-\alpha)\,b\,\mathbf 1.$$

Khi train, thêm nhiễu pixel $\mathcal N(0,0.02^2)$. Khi giải Δ-CE thì không thêm nhiễu.

**Nhãn:** $y=\mathbb 1[\phi_1>0.5]$, với 5% nhiễu nhãn trong tập train.

**Kích thước tập:** train 20 000, val 5 000, test 5 000 (cố định theo seed dữ liệu 1234).

### 4.3. Tier 1 — Colored MorphoMNIST (4 vs 9)

**Nguồn:** các chữ số 4 và 9 của MNIST (torchvision); dùng split train/test gốc, tách 10% train làm val.

**Độ dày nét:** chọn độ dày mục tiêu $t\sim U[1.5,5.0]$ px. Làm dày hoặc mỏng nét bằng `skimage.morphology` (dilation/erosion với `disk(1)`, lặp cho tới gần $t$ nhất). Đo lại độ dày thực tế bằng **$2\times$ trung bình distance transform tại các pixel thuộc skeleton** (`skimage.morphology.skeletonize`, `scipy.ndimage.distance_transform_edt`). **Giá trị đo được này là ground truth.**

**Màu:**
- $\phi_c\sim U[0,1]$, màu nét $c=(1-\phi_c)(1,0,0)+\phi_c(0,0,1)$;
- nền xám $b=0.1+0.5\,\phi_b$, $\phi_b\sim U[0,1]$;
- $I=\alpha c+(1-\alpha)b\mathbf 1$, với $\alpha$ = ảnh xám đã chuẩn hóa về $[0,1]$.

**Factor:**

| Factor | Đo bằng |
|---|---|
| shape | shape oracle (§5.3) |
| color | $\phi_c$ |
| bg | $\phi_b$ |
| thick | độ dày đã chuẩn hóa |

### 4.4. Các thay đổi được tiêm vào (drift → adaptation)

**Quy trình chung:**
- $f_{old}$ được train trên dữ liệu sạch.
- $f_{new}$ = $f_{old}$ fine-tune trên **tập adaptation** 10 000 mẫu (Tier 0) hoặc 8 000 mẫu (Tier 1).
- Fine-tune: Adam, lr $3\cdot10^{-4}$, batch 256, 3 epoch.

| Mã | Loại drift | Tập adaptation | $k^\*$ | Tier |
|---|---|---|---|---|
| **T1** | Spurious correlation drift | Tier 0: 70% mẫu có $\phi_1\sim U[0.4,0.6]$ (vùng mơ hồ), phần còn lại lấy như dữ liệu sạch; với xác suất 0.95, $\phi_3\sim U[0.5,1]$ nếu $y=1$ và $U[0,0.5]$ nếu $y=0$. Tier 1: tương quan màu–nhãn 0.95 | color | 0, 1 |
| **T3** | Real drift trên một nhóm con (phi cục bộ) | Đảo nhãn khi $\phi_2>0.75$ (Tier 0), hoặc khi độ dày > quantile 0.85 (Tier 1) | size / thick | 0, 1 |
| **T4** | Real drift: dịch ngưỡng | $y=\mathbb 1[\phi_1>0.6]$ | aspect | 0 |
| **N0** | Không drift | Dữ liệu sạch, khác seed thứ tự dữ liệu | — | 0, 1 |
| **N1** | Virtual drift | $\phi_2\sim U[0.5,1]$, quy tắc nhãn giữ nguyên | — (kỳ vọng không đổi hành vi) | 0 |

**Số cặp mô hình:**
- Tier 0: T1, T3, T4 mỗi loại 3 seed (0–2); N0 có 5 seed (0–4); N1 có 3 seed (0–2). Tổng **17 cặp**.
- Tier 1: T1, T3 mỗi loại 3 seed; N0 có 5 seed. Tổng **11 cặp**.
- Mỗi seed $s$ có một $f_{old}^{(s)}$ riêng, dùng chung cho mọi điều kiện của seed đó.

**Núm vặn cho G0** (khai báo trước, tối đa 2 lần chỉnh mỗi điều kiện, ghi log):
- T1: tỉ lệ vùng mơ hồ 0.7 → 0.9; tương quan 0.95 → 0.99; số epoch 3 → 5.
- T3, T4: số epoch 3 → 5.
- T3 Tier 1: ngưỡng quantile 0.85 → 0.80.

---

## 5. Mô hình

### 5.1. Classifier (dùng chung cho mọi tier)

- **Kiến trúc:** Conv(3→32, 3×3, pad 1)–ReLU–MaxPool2 → Conv(32→64)–ReLU–MaxPool2 → Conv(64→64)–ReLU–GlobalAvgPool → Linear(64→2). Khoảng 60k tham số.
- **Train $f_{old}$:** Adam, lr $10^{-3}$, batch 256, 15 epoch (Tier 0) / 10 epoch (Tier 1). Lưu checkpoint epoch cuối.
- **Yêu cầu:** accuracy trên test sạch ≥ 95% (Tier 0) / ≥ 97% (Tier 1).

### 5.2. VAE (Tier 1)

- **Encoder:** Conv(3→32, s2)–Conv(32→64, s2)–FC(→256)–(μ, logσ²) với dim 16.
- **Decoder:** đối xứng, đầu ra sigmoid. Loss: BCE + KL ($\beta=1$).
- Train trên dữ liệu **sạch**, 30 epoch, Adam lr $10^{-3}$.
- **Mã hóa điểm đánh giá:** $z_0=\mu(x)$. Mọi so sánh factor dùng $D(z_0)$, không dùng $x$ gốc.

### 5.3. Oracle Tier 1 (giải tích, không học trừ shape)

Vì màu đối tượng không có thành phần G, mỗi pixel thỏa:
$$I_R-I_G=\alpha(1-c),\qquad I_B-I_G=\alpha c,\qquad I_G=(1-\alpha)b.$$

Từ đó:
- $\hat\alpha_p=\mathrm{clip}(I_{R}+I_{B}-2I_{G},0,1)$;
- $\hat c=\dfrac{\sum_p\hat\alpha_p\,\mathrm{clip}\big((I_B-I_G)/\hat\alpha_p,0,1\big)}{\sum_p\hat\alpha_p}$, chỉ lấy các pixel có $\hat\alpha_p>0.2$;
- $\hat b=\mathrm{median}\{I_{G,p}/(1-\hat\alpha_p):\hat\alpha_p<0.05\}$;
- $\widehat{\text{thick}}$ = phép đo độ dày ở §4.3 áp dụng cho mask $\hat\alpha>0.5$;
- **shape oracle:** classifier ở §5.1 nhưng nhận 1 kênh $\hat\alpha$, train trên $\hat\alpha$ của dữ liệu sạch với mọi độ dày. Đầu ra $\widehat{\text{shape}}=p(9)$.

**Chuẩn hóa:** $\tilde\phi_k=\hat\phi_k/\sigma_k$, với $\sigma_k$ là độ lệch chuẩn của $\hat\phi_k$ trên $D(\mu(x))$ của tập test sạch.

**Tier 0:** $\phi$ chính xác, $\sigma_k=1$.

---

## 6. Thuật toán

### 6.1. Bộ giải Δ-CE (`solvers/dce.py`)

```
input: z0, m_o, m_n, kappa=0.5, r, R=5, T=400, lr=0.05, lam0=1.0, mu, sigma=0.05
best = NOT_FOUND
for j in 1..R:                               # restart j=1 không có nhiễu
    z = z0 + (j>1)*sigma*eps_j ; lam = lam0
    for t in 1..T:
        L = relu(m_n(z)+kappa) + relu(kappa-m_o(z)) + lam*||z-z0||^2 + mu*Reg(z)
        z = AdamStep(z, grad L)
        Tier0: z = clamp(z, 0, 1)            # projected gradient
        if m_n(z) <= -kappa and m_o(z) >= kappa:   # kiểm tra mỗi bước, theo batch
            if d(z0,z) < d(best): best = z
        if t == T/2 and best is NOT_FOUND: lam = lam/2
return best if d(z0,best) <= r else NOT_FOUND
```

**Thông số:**
- Tier 0: $\mu=0$ (đã có clamp). Tier 1: $\mathrm{Reg}(z)=\|z\|^2$, $\mu=0.01$.
- Chạy theo batch 300 điểm × R restart trên GPU, hoặc trên CPU cho Tier 0.
- Seed nhiễu restart: `seed_restart = 10_000 + pair_id`.

### 6.2. CE thường (`solvers/ordinary_ce.py`)

Giống §6.1 nhưng chỉ có một mô hình: loss là $\mathrm{relu}(m(z)+\kappa)$ + proximity + Reg. Được dùng cho:
- B1, B1-det;
- ablation A2 (CE thường của $f_{new}$);
- hiệu chuẩn $r$.

### 6.3. Hiệu chuẩn ngân sách $r$ (làm một lần, đóng băng)

$r_{\text{tier}}=2\times\mathrm{median}_i\, d(z_{0,i}, \mathrm{CE}(z_{0,i}; f_{new}))$, tính trên 300 điểm validation của cặp **T1 seed 0** ở mỗi tier. Sau đó ghi vào config và không đổi nữa.

### 6.4. Baseline

| Mã | Định nghĩa chính xác | Vector giải thích $\Delta\tilde\phi$ |
|---|---|---|
| **B1** (Artelt Định nghĩa 3) | $z'_o=\mathrm{CE}(z_0;f_{old})$ và $z'_n=\mathrm{CE}(z_0;f_{new})$, giải **độc lập** với $R=5$ restart, seed restart khác nhau (`seed+0` / `seed+1`) | $\tilde\phi(D(z'_n))-\tilde\phi(D(z'_o))$. Hợp lệ khi cả hai CE hợp lệ và $d\le r$ |
| **B1-det** | Như B1 nhưng $R=1$, không nhiễu, cùng khởi tạo $z_0$ | như B1 |
| **B3** (Artelt Eq. 14, xấp xỉ gradient) | $g=\nabla_z m_n/\|\nabla_z m_n\|-\nabla_z m_o/\|\nabla_z m_o\|$ tại $z_0$ | Tier 0: $\Delta\tilde\phi=-g$ (đã ở không gian factor). Tier 1: $\tilde\phi(D(z_0-\tfrac r2\,g/\|g\|))-\tilde\phi(D(z_0))$. Luôn hợp lệ, trừ khi $\|g\|<10^{-8}$ |
| **B4** (hướng Muschalik; chỉ Tier 0, toàn cục) | Hiệu permutation importance theo factor giữa hai mô hình (2000 điểm test, 5 lần hoán vị) | Một vector dùng chung cho mọi điểm |

**Ablation:**
- **A1:** $\lambda=0$ (tối đa bất đồng không cần gần $x$, tương tự DiG-IN).
- **A2:** bỏ ràng buộc $m_o\ge\kappa$.
- **A3:** $\kappa\in\{0,0.5,1.0\}$.

Ablation chạy đủ ở Tier 0; ở Tier 1 chỉ chạy seed 0.

---

## 7. Metric

Với điểm $i$ và phương pháp $M$:
- $v_i\in\{0,1\}$ cho biết có giải thích hợp lệ hay không;
- $\Delta\tilde\phi_i$ là vector giải thích;
- $\mathrm{top}_i=\arg\max_k|\Delta\tilde\phi_{ik}|$.

| Metric | Công thức | Vai trò |
|---|---|---|
| **Recovery@r** | $\frac1N\sum_i v_i\,\mathbb 1[\mathrm{top}_i=k^\*]$ (trường hợp không hợp lệ hoặc `NOT_FOUND` tính là 0) | **H1, H2** |
| Share | $\mathrm{mean}_{i:v_i=1}\ \lvert\Delta\tilde\phi_{ik^\*}\rvert/\sum_k\lvert\Delta\tilde\phi_{ik}\rvert$ | Phụ |
| Found-rate | $\frac1N\sum_i v_i$ | Báo cáo |
| **Điểm thay đổi mức cặp** $S(M,\text{pair})$ | Δ-CE: found-rate (loại N ∪ loại O). B1, B1-det, B3: $\mathrm{median}_i\|\Delta\tilde\phi_i\|_2$ | **H3** (AUROC) |
| SAR | Trên các cặp N0, N1: tỉ lệ cặp có $\max_k\mathrm{mean}_{i:v_i=1}\mathrm{share}_{ik}\ge0.5$ **và** found-rate ≥ 0.2 | Phụ |
| Plausibility (Tier 1) | Tỉ lệ $D(z^\*)$ mà shape oracle tin cậy ≥ 0.9 vào một trong hai lớp; khoảng cách 5-NN trong không gian feature penultimate của shape oracle tới dữ liệu test thật | Báo cáo |
| Runtime | Giây/điểm (GPU hoặc CPU ghi rõ) | Báo cáo |

**Điểm đánh giá:** $N=300$ (Tier 0) hoặc $N=200$ (Tier 1) điểm hợp lệ (§3.2) trong tập test, lấy mẫu với seed `5000+pair_id`. Dùng cùng một tập điểm cho mọi phương pháp của cùng một cặp.

---

## 8. Giả thuyết và thống kê (khai báo trước)

| ID | Phát biểu | Kiểm định | Tiêu chí đạt |
|---|---|---|---|
| **H4** | Bộ giải khớp lý thuyết tuyến tính | Tier L, 1000 cặp | Sai số tương đối của $\delta_\Delta$ < $10^{-3}$; $\lvert\cos(Pu,P\Delta w)\rvert$ > 0.999 ở ≥ 99% cặp |
| **H1** | Δ-CE > B1 (Tier 1, gộp T1 + T3) | Hiệu Recovery@r | CI 95% > 0 sau Holm **và** cùng dấu ở 3/3 seed |
| **H2a** | Δ-CE > B3 trên T3, Tier 0 | như trên | như trên |
| **H2b** | Δ-CE > B3 trên T3, Tier 1 | như trên | như trên |
| **H3** | Δ-CE phân biệt được cặp có drift thật (T*) với cặp null (N0, N1) | AUROC mức cặp của $S$ | AUROC(Δ-CE) ≥ 0.90 ở mỗi tier. So sánh với B1, B3 chỉ mang tính mô tả |
| **P-tie** (dự đoán, không phải kiểm định) | Trên T1, T4 Tier 0: Δ-CE ≈ B3 ≈ B1-det | $\lvert$hiệu Recovery$\rvert$ | ≤ 0.10 thì coi là xác nhận Mệnh đề 4 |

**Thống kê:**
- **CI:** paired bootstrap 10 000 lần, lấy lại điểm **phân tầng theo cặp**, rồi lấy trung bình qua các cặp.
- **Holm:** áp dụng trên ba kiểm định chính H1, H2a, H2b.
- **AUROC** với số cặp nhỏ (Tier 0: 9 cặp drift so với 8 cặp null; Tier 1: 6 so với 5): báo cáo giá trị chính xác. Không tính CI.
- Bảng phụ: mọi kết quả theo từng seed.

---

## 9. Gate (rẻ trước, đắt sau)

| Gate | Hạn | Nội dung | Đạt khi | Nếu trượt |
|---|---|---|---|---|
| **G-lit** | 17/10 | Đọc full text Artelt et al. NPL 2022 (phần persistent explanations); Hammer & Hüllermeier 2021; Muschalik et al. 2022; Hinder et al. "Model-based explanations of concept drift"; tra Semantic Scholar các bài trích dẫn Artelt 2021 và DiG-IN | Không có công trình định nghĩa witness tối thiểu trên vùng bất đồng kèm benchmark ground truth | Chuyển trọng tâm sang benchmark + H3; ghi log |
| **G0** (hiện tượng) | 20/10 | Tier 0, CPU | Xem bảng G0 dưới | Dùng núm vặn §4.4 (tối đa 2 lần). Vẫn trượt thì báo cáo lại và chọn Dừng hay thu hẹp |
| **G1** | 21/10 | H4; pilot Tier 0 với seed 0, $N=100$, mọi phương pháp; đo runtime | H4 đạt; found-rate Δ-CE trên T1 ≥ 0.6; runtime ≤ 5 phút mỗi cặp | Sửa bộ giải (restart, lịch $\lambda$, khởi tạo từ CE thường) |
| **G2** | 24/10 | Tier 1: VAE, oracle, $f_{old}$ | Spearman($\hat\phi$ trên $D(\mu(x))$, $\phi$ thật) ≥ 0.9 cho color và bg, ≥ 0.8 cho thick; shape oracle acc ≥ 98% trên ảnh thật và ≥ 95% trên ảnh tái tạo; G0 cho Tier 1 đạt | thick trượt: thay T3-Tier1 bằng đảo nhãn theo bg > q85 (khai báo trước ở đây) |
| **G3** | 1/11 | Chạy đủ seed, kiểm định | — | Chọn route (§10) |

**Tiêu chí G0 chi tiết.** Đo trên 2000 điểm test sạch. $\mathrm{Flip}_k(f)$ = tỉ lệ dự đoán đổi khi lấy mẫu lại factor $k$ (giữ nguyên các factor khác).

| Điều kiện | Tiêu chí |
|---|---|
| T1 | $\mathrm{Flip}_{color}(f_{new})-\mathrm{Flip}_{color}(f_{old})\ge0.15$ |
| T3 | Accuracy của $f_{new}$ theo nhãn mới trên nhóm con ≥ 0.85; tỉ lệ $f_{new}=f_{old}$ ngoài nhóm con ≥ 0.95 |
| T4 | Accuracy của $f_{new}$ theo nhãn mới trên dải $\phi_1\in(0.5,0.6)$ ≥ 0.80 |
| N0, N1 | Tỉ lệ $f_{new}=f_{old}$ trên test ≥ 0.98; $\lvert\Delta\mathrm{Flip}_k\rvert\le0.05$ với mọi $k$ |
| Mọi điều kiện | $f_{old}$ đạt accuracy yêu cầu ở §5.1 |

---

## 10. Route (khai báo trước)

| Route | Điều kiện | Câu chuyện paper (6 trang) |
|---|---|---|
| **A** | H4, H3 đạt; ít nhất một trong H1/H2a/H2b đạt | *Differential counterfactuals: minimal witnesses of changing decision behaviour* — gồm định nghĩa, L1–P4, benchmark, kết quả |
| **B** | H4 đạt; H1, H2a, H2b đều trượt; H3 đạt | *When do counterfactuals explain model adaptation?* — lý thuyết, cộng bằng chứng rằng xấp xỉ gradient / hiệu CE đủ dùng với thay đổi cục bộ, cộng khả năng phát hiện mức cặp của witness |
| **B′** | H3 cũng trượt | Paper benchmark: *Ground-truth benchmark for explaining model adaptation* — gồm thiết kế drift tiêm vào, các metric, kết quả âm có kiểm soát. Báo cáo lại trước khi viết |
| **Dừng** | G0 trượt sau 2 lần chỉnh, hoặc G-lit phát hiện trùng hoàn toàn | — |

---

## 11. Ma trận chạy và ngân sách tính toán

| Khối | Số lượng | Ước tính |
|---|---|---|
| Tier L | 1000 cặp × 2 chiều | < 1 phút CPU |
| Tier 0: $f_{old}$ × 5 seed; fine-tune × 17 | ~1 phút mỗi lượt train trên CPU-GPU nhỏ | ~0.5 giờ CPU |
| Tier 0: Δ-CE (N + O), B1, B1-det, B3, B4, A1–A3 × 17 cặp × 300 điểm | | ~1–2 giờ CPU, hoặc ~0.3 GPU-giờ |
| Tier 1: VAE + shape oracle + $f_{old}$ × 5 + fine-tune × 11 | | ~0.7 GPU-giờ |
| Tier 1: các phương pháp × 11 cặp × 200 điểm (+ A1–A3 seed 0) | | ~1.5 GPU-giờ |
| Dự phòng chạy lại | | ~2 GPU-giờ |
| **Tổng** | | **~4.5 GPU-giờ; trần 10** |

**Quy tắc vận hành:**
1. Tier L và Tier 0 chạy local trên CPU.
2. Tier 1 gộp thành **≤ 4 job**: (a) dữ liệu + VAE + oracle; (b) classifier; (c) phương pháp T1 + T3; (d) N0 + ablation. Mỗi job < 60 phút và có checkpoint.
3. Mọi script idempotent: bỏ qua nếu đã có output với cùng hash config.
4. Log JSON cho mỗi cặp (`results/{tier}/{cond}_s{seed}.json`).

---

## 12. Cấu trúc repo

```
dce/
  configs/{tierL,tier0,tier1}.yaml        # mọi siêu tham số ở §4–8
  data/renderer.py  data/morpho_color.py  data/drift.py      # §4.2–4.4
  models/cnn.py  models/vae.py
  oracle/tier1_oracle.py  oracle/shape_oracle.py             # §5.3
  solvers/dce.py  solvers/ordinary_ce.py                     # §6.1–6.2
  baselines/b1_diff.py  baselines/b3_grad.py  baselines/b4_perm.py
  eval/metrics.py  eval/stats.py                             # §7–8
  scripts/g0_phenomenon.py  scripts/calibrate_r.py
  scripts/run_tier.py  scripts/make_tables.py  scripts/make_figure.py
  tests/test_linear_theory.py              # H4, chạy trong CI local
  logs/decisions.md                        # §13
  results/
```

---

## 13. Kiểm soát thay đổi sau khi khóa spec

- Mỗi thay đổi ghi theo mẫu: `D-xxx | ngày | mục spec | thay đổi | lý do | trước/sau khi thấy kết quả`.
- Thay đổi **sau khi đã thấy kết quả** của một kiểm định chính thì không được dùng cho kiểm định đó. Chỉ được báo cáo là phân tích khám phá.
- Các núm vặn G0 (§4.4) và phương án thay thế ở G2 đã được khai báo trước, nên không tính là vi phạm.

---

## 14. Kế hoạch paper (6 trang **gồm** hình và references; không appendix)

Template `esannV2.cls`, khổ A4, single-blind. Đánh dấu special session trên form nộp bài.

| Phần | Trang | Nội dung |
|---|---|---|
| Tiêu đề + abstract | 0.35 | — |
| 1 Introduction + related work | 0.9 | Model adaptation sau drift; Artelt et al. (định vị theo §2); robust CE under model change là bài toán đối ngẫu; DiG-IN; Hinder & Hammer |
| 2 Differential counterfactuals | 1.3 | Định nghĩa; L1 và P2 kèm chứng minh một dòng; P3/P4 kèm phác thảo và cảnh báo; link tới chứng minh đầy đủ |
| 3 Benchmark with injected drift | 0.8 | Tier 0/1, bảng drift (T1/T3/T4/N0/N1), metric |
| 4 Experiments | 1.6 | **Bảng 1:** Recovery@r (phương pháp × điều kiện × tier). **Hình 1:** witness Tier 0 trong không gian factor và ví dụ Tier 1. AUROC (H3) trong văn bản hoặc một bảng nhỏ |
| 5 Discussion & conclusion | 0.3 | Hành vi vs cơ chế; chứng chỉ không chặt; dữ liệu tổng hợp |
| References | 0.75 | ~14 tài liệu |

**Danh sách tham khảo tối thiểu:** Artelt et al. 2021; Artelt et al. NPL 2022; Hinder & Hammer 2020; Hammer & Hüllermeier 2021; Muschalik et al. 2022; Wachter et al. 2017; Hamman et al. ICML 2023; DiG-IN (CVPR 2024); FastDiME (ECCV 2024); ECCCo (AAAI 2024); Pawelczyk et al. UAI 2020; Castro et al. JMLR 2019 (Morpho-MNIST); Kaden et al. ESANN 2026; Yan et al. CVPR 2021 (negative flips, **tra lại trước khi trích**).

---

## 15. Timeline

| Ngày | Việc | Mốc |
|---|---|---|
| 15–17/10 | G-lit; viết `renderer.py`, `drift.py`, `cnn.py`, `test_linear_theory.py`; review spec | **Khóa spec 17/10** |
| 18–20/10 | Tier 0: train, G0; cài `dce.py`, `ordinary_ce.py` | **G0 20/10** |
| 21/10 | H4 + pilot Tier 0 + hiệu chuẩn $r$ | **G1** |
| 22–24/10 | Tier 1: dữ liệu, VAE, oracle, shape oracle, $f_{old}$, G0-Tier1 | **G2 24/10** |
| 25–31/10 | Chạy đủ ma trận ở hai tier, ablation | — |
| 1/11 | Thống kê, chọn route | **G3** |
| 2–8/11 | Viết paper 6 trang, hình, bảng; đẩy chứng minh đầy đủ lên GitHub | Bản nháp 8/11 |
| 9–12/11 | Kiểm lại chứng minh và số liệu; chạy lại một cặp từ đầu để tái lập | — |
| 13–16/11 | Chỉnh sửa, kiểm khổ A4 và lề, nộp và chọn special session | **Nộp 16/11** |
| 17–18/11 | Dự phòng | Hạn cứng 18/11 |

---

## 16. Việc cần xác nhận trước khi khóa spec

1. Chọn special session *From Understanding Distributional Change to Informed Action* hay track thường.
2. Cam kết của tác giả: nếu bài được nhận, ít nhất một tác giả phải đăng ký và trả phí trước 12/02/2027 và trình bày tại hội nghị.
3. Tài khoản/máy chạy Tier 1 (≤ 4 job GPU) và quota còn lại.
4. Có công bố preprint arXiv sau khi nộp hay không. Được phép nếu ghi rõ là bài nộp ESANN, nhưng tác giả tự chịu rủi ro về tính mới.
