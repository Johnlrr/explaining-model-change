# Proposal: Differential Counterfactual Explanations (Δ-CE) — giải thích sự thay đổi của mô hình bằng witness tối thiểu

**Venue mục tiêu:** ESANN 2027 (hạn nộp 18/11/2026; nộp nội bộ 16/11/2026)
**Phiên bản:** v1 — 15/10/2026
**Trạng thái:** trước pilot. Khóa spec ngày 17/10/2026.
**Ngân sách tính toán:** trần 12 GPU-giờ, ước tính ~6 GPU-giờ (chưa đo). Tier 0 chạy được hoàn toàn trên CPU.

---

## 0. Tóm tắt một đoạn

Khi một classifier được cập nhật từ $f_{old}$ sang $f_{new}$, người dùng cần biết *mô hình đã thay đổi cái gì*. Các phương pháp hiện có giải thích model change theo ba cách: so sánh hai CE tính độc lập (Artelt et al., chỉ cho mô hình tuyến tính hoặc tabular), so sánh feature importance, hoặc sinh ảnh tối đa hóa bất đồng một cách toàn cục (DiG-IN).

Chúng tôi định nghĩa **Δ-CE**: thay đổi tối thiểu trên data manifold khiến **đúng một** trong hai mô hình đổi quyết định. Một Δ-CE là một **witness**: một input cụ thể, kiểm chứng được, mà trên đó hai mô hình bất đồng.

Đóng góp gồm ba phần:
- **(i) Lý thuyết.** Bổ đề "band containment" cho mọi Δ-CE; cận dưới khoảng cách cho ra chứng chỉ "không có thay đổi" (null); nghiệm đóng và kết quả khôi phục hướng thay đổi trong trường hợp tuyến tính.
- **(ii) Thuật toán.** Giải Δ-CE trên manifold, gồm không gian factor của một renderer khả vi và latent của VAE.
- **(iii) Benchmark có ground truth.** Tiêm các thay đổi đã biết (thêm shortcut, đổi ngưỡng quyết định, đảo nhãn một nhóm con) và một điều kiện null, đo bằng oracle giải tích, **không dùng auditor học từ nhãn người**.

---

## 1. Bài học từ đề tài trước và cách proposal này xử lý

| Nguyên nhân thất bại trước | Biện pháp trong proposal này |
|---|---|
| Gate đặt sai thứ tự: tốn công kiểm chứng công cụ đánh giá trước khi biết hiện tượng có tồn tại | **G0 là kiểm tra hiện tượng**, chạy CPU trong ≤ 2 ngày, trước mọi việc khác |
| Auditor cần ≥ 200 panel nhãn độc lập | **Không có auditor học từ người.** Tier 0 dùng factor đúng tuyệt đối (renderer tổng hợp). Tier 1 dùng oracle giải tích (màu, độ sáng nền, độ dày nét), được hiệu chuẩn tự động bằng ground truth |
| Panel sinh ra lỗi (mask, ghép cắt, diffusion) | Không dùng diffusion hay inpainting. Tier 0 có manifold chính xác; Tier 1 dùng VAE nhỏ trên dữ liệu 28×28 |
| Lỗi hạ tầng tốn lượt submit | Tier 0 chạy local trên CPU. Mỗi job GPU < 60 phút, có checkpoint. Gộp nhiều cấu hình vào một job |
| Ngưỡng thành công cao, novelty "uncertain" | Kiểm định chính là CI 95% không chứa 0. Kích thước hiệu ứng báo cáo riêng. **Các trường hợp hòa đã được lý thuyết dự đoán trước, nên không bị tính là thất bại** (xem §5.3). Route B được khai báo trước |
| Ước tính GPU vượt ngân sách | Mô hình nhỏ (~100k tham số). Tổng ước tính ~6 GPU-giờ, trần 12. Đo thời gian thật ở G0 |

---

## 2. Kiểm tra novelty (15/10/2026)

### 2.1. Các công trình gần nhất và khác biệt

| Công trình | Làm gì | Khác Δ-CE ở đâu |
|---|---|---|
| Artelt et al. 2021 ([arXiv 2104.02459](https://arxiv.org/pdf/2104.02459)); Artelt et al., *Neural Processing Letters* 2022 ([link](https://link.springer.com/article/10.1007/s11063-022-10826-5)) | Giải thích model adaptation bằng *so sánh hai CE* tính độc lập; định lý cho **mô hình tuyến tính** | Δ-CE giải *một* bài toán trên vùng bất đồng, cho deep model, trên manifold, có chứng chỉ null. Baseline B1 của chúng tôi chính là cách tiếp cận này |
| Hinder & Hammer 2020, *Counterfactual Explanations of Concept Drift* ([arXiv 2006.12822](https://arxiv.org/pdf/2006.12822)) | Giải thích **drift của phân phối dữ liệu** bằng CE | Δ-CE giải thích thay đổi **của mô hình**, không phải của dữ liệu |
| Muschalik et al. (*Agnostic explanation of model change based on feature importance*); Hammer & Hüllermeier 2021 | Giải thích model change bằng feature importance; nêu bài toán ở mức khái niệm | Không sinh witness cục bộ; không làm cho ảnh |
| Group CE cho drift (2025, [arXiv 2509.09616](https://arxiv.org/html/2509.09616v1)) | Theo dõi sự tiến hóa của group CE theo thời gian | Vẫn ở dạng so sánh CE qua thời gian, cho tabular |
| DiG-IN (CVPR 2024, [arXiv 2311.17833](https://arxiv.org/pdf/2311.17833)) | Diffusion guidance sinh ảnh **tối đa hóa bất đồng** giữa classifier | Không neo vào một input $x$, không có ràng buộc tối thiểu, không gắn với cập nhật mô hình. Tương ứng ablation A1 (bỏ proximity) |
| Robust CE under model change (Hamman et al. ICML 2023; [PSCE arXiv 2601.16659](https://arxiv.org/pdf/2601.16659v2); [AIJ 2025](https://www.sciencedirect.com/science/article/pii/S000437022500178X)) | Tìm CE **hợp lệ cho cả hai** mô hình | **Mục tiêu ngược lại**: Δ-CE tìm điểm mà hai mô hình *bất đồng*. Có thể trình bày Δ-CE như bài toán đối ngẫu của robust CE |
| FastDiME (ECCV 2024, [arXiv 2312.14223](https://arxiv.org/abs/2312.14223v2)) | Dùng CE để phát hiện shortcut của **một** mô hình | Δ-CE phát hiện *thay đổi* giữa hai mô hình, kể cả thay đổi không phải shortcut (T4 đổi ngưỡng) |
| *A Geometric View of Counterfactual Behavior* (2026, [arXiv 2606.04209](https://arxiv.org/pdf/2606.04209)) | Hai mô hình cùng accuracy có thể khác hẳn về hành vi CE | Dùng làm **động lực** cho bài toán; không đề xuất giải thích sự khác biệt |
| ECCCo (AAAI 2024, [link](https://ojs.aaai.org/index.php/AAAI/article/view/28956)) | Faithfulness của CE với *một* mô hình | Không xét hai mô hình |

**Bối cảnh venue.** ESANN 2026 có special session *"Reliable counterfactuals for machine learning models"* (Kaden, Paassen, Hammer, Villmann) ([proceedings](https://www.esann.org/proceedings/2026)). Nhóm Bielefeld (Hammer) có cả một dòng nghiên cứu về giải thích model change và drift. Đây đúng là nhóm reviewer quan tâm đến bài toán này.

### 2.2. Claim được phép và không được phép

**Được claim (sau khi đọc full text ở bước W0):**
> We formalise differential counterfactuals — minimal on-manifold witnesses on which exactly one of two models changes its decision — and show that (a) they come with a distance-based certificate of local model agreement, (b) in the linear case they recover the projected parameter change, and (c) on controlled benchmarks with injected model changes they recover non-local changes that gradient-difference baselines miss, while counterfactual differencing produces systematic false attributions under non-unique CE generators.

**Không được claim:**
- "First to explain model change with counterfactuals": Artelt et al. và Hinder & Hammer đã làm.
- "Δ-CE recovers change direction better than CE differencing in general": sai trong trường hợp tuyến tính (Mệnh đề 4).
- Chứng chỉ null cho deep net trên thực tế: cần hằng số Lipschitz của $h$, không tính được chính xác (§3.6).

### 2.3. Rủi ro novelty còn lại

Mức độ: **trung bình–thấp**. Tìm kiếm của chúng tôi dựa trên web search, không phải Google Scholar full-text. Việc cần làm ở W0 (15–17/10):
1. Đọc full text Artelt et al. NPL 2022 (mục "persistent explanations" và các định lý).
2. Đọc Hammer & Hüllermeier 2021 và Muschalik et al.
3. Tra các bài trích dẫn Artelt 2021 và DiG-IN trên Semantic Scholar.
4. Tra thêm các từ khóa: "explaining model updates", "negative flip explanation", "backward-compatible explanations", "explanation of model differences counterfactual", "disagreement witness".

**Quy tắc dừng:** nếu tìm thấy công trình định nghĩa CE trên vùng bất đồng cho deep model và có benchmark ground truth, thì chuyển trọng tâm sang benchmark và chứng chỉ null (§3.3) và báo cáo lại trước ngày 18/10.

---

## 3. Lý thuyết

### 3.1. Thiết lập

- $(\mathcal Z, d)$ là không gian metric chứa ứng viên. Đây có thể là không gian factor đã chuẩn hóa (Tier 0) hoặc latent của VAE (Tier 1). Ảnh là $D(z)$ với decoder hoặc renderer $D$.
- Hai classifier có logit $f_{old}, f_{new}$. Xét các $x$ mà **cả hai dự đoán cùng lớp $y$**.
- Margin lớp $y$: $m(z) = f_y(D(z)) - \max_{j\neq y} f_j(D(z))$. Ta có $m(z)>0 \iff$ mô hình dự đoán $y$ tại $z$, với quy ước tie-break không ưu tiên $y$.
- Ký hiệu $m_o, m_n$ cho margin của hai mô hình, và $h := m_n - m_o$ là **hàm thay đổi**.
- Biên quyết định $\kappa \ge 0$ (yêu cầu dự đoán đủ dứt khoát).

**Định nghĩa 1 (Δ-CE loại N, "chỉ mô hình mới đổi").** Với $z_0$ là biểu diễn của $x$, sao cho $m_o(z_0)\ge\kappa$ và $m_n(z_0)>0$:
$$
\mathcal F_\kappa=\{z\in\mathcal Z:\ m_n(z)\le-\kappa,\ \ m_o(z)\ge\kappa\},\qquad
\delta_\Delta(z_0)=\inf_{z\in\mathcal F_\kappa} d(z_0,z)\quad(\inf\emptyset=+\infty),
$$
và Δ-CE là điểm $z^\*$ đạt (hoặc xấp xỉ) $\delta_\Delta$.

Loại O ("chỉ mô hình cũ đổi") định nghĩa đối xứng bằng cách hoán đổi $m_o$ và $m_n$.

**Tính chất witness.** Mọi $z\in\mathcal F_\kappa$ thỏa $f_{old}(D(z)) = y \ne f_{new}(D(z))$, kiểm chứng được bằng hai lần forward pass. Hiệu của hai CE độc lập, $x'_{new}-x'_{old}$, **không** có tính chất này: nó không phải một input mà hai mô hình bất đồng.

### 3.2. Bổ đề 1 (band containment) — đã chứng minh

> Với mọi $z\in\mathcal F_\kappa$: $\ h(z)\le-2\kappa\ $ và $\ |h(z)|\ \ge\ m_o(z)+\kappa\ \ge 2\kappa$.

*Chứng minh.* $h(z)=m_n(z)-m_o(z)\le-\kappa-m_o(z)\le-2\kappa$. Vì $h(z)<0$ nên $|h(z)|=-h(z)\ge \kappa+m_o(z)$. ∎

**Ý nghĩa.** Witness chỉ tồn tại ở nơi độ thay đổi $|h|$ vượt quá margin của mô hình cũ. Nếu ở mọi nơi hai mô hình khác nhau ít hơn $2\kappa$ (theo margin) thì $\mathcal F_\kappa=\emptyset$. Kết quả này đúng với **mọi** hàm $m_o, m_n$; không cần tuyến tính hay khả vi.

### 3.3. Mệnh đề 2 (cận dưới khoảng cách, chứng chỉ null cục bộ) — đã chứng minh

> (a) Nếu $m_n$ là $L_n$-Lipschitz trên $B(z_0,r)$ thì không có Δ-CE trong $B\big(z_0,\min\{r,\ (m_n(z_0)+\kappa)/L_n\}\big)$.
>
> (b) Nếu $h$ là $L_h$-Lipschitz trên $B(z_0,r)$ và $h(z_0)+2\kappa>0$ thì không có Δ-CE trong $B\big(z_0,\min\{r,\ (h(z_0)+2\kappa)/L_h\}\big)$.

*Chứng minh (b).* Giả sử $d(z_0,z)<(h(z_0)+2\kappa)/L_h$ và $z \in B(z_0,r)$. Khi đó $h(z)\ge h(z_0)-L_h d(z_0,z)>-2\kappa$, nên theo Bổ đề 1 ta có $z\notin\mathcal F_\kappa$. Phần (a) chứng minh tương tự, vì $m_n$ phải giảm ít nhất $m_n(z_0)+\kappa$. ∎

**Ý nghĩa.** Phần (a) chính là cận của CE thông thường. Phần (b) là **cận riêng của Δ-CE**: nếu mô hình thay đổi ít ($L_h$ nhỏ) và hai mô hình đồng ý tại $x$ ($h(z_0)\approx 0$), thì witness nằm xa ít nhất $\approx 2\kappa/L_h$. Khi $f_{old}\equiv f_{new}$ ($L_h=0$, $h\equiv0$) và $\kappa>0$, không tồn tại witness ở đâu cả. Đây là cơ sở cho hành vi null: **với ngân sách khoảng cách $r$ hữu hạn, Δ-CE trả về "không tìm thấy" khi mô hình không đổi**.

**Giới hạn.** Với deep net, $L_h$ không tính được chính xác. Ta chỉ ước lượng cục bộ bằng $\max\|\nabla_z h\|$ trên mẫu, nên đây **không phải chứng chỉ chặt**. Trong paper, phần này chỉ được dùng làm cơ sở lý thuyết cho thiết kế và giải thích kết quả null; không claim certified.

### 3.4. Mệnh đề 3 (nghiệm đóng, trường hợp tuyến tính) — đã chứng minh

Giả sử $\mathcal Z=\mathbb R^p$ với metric Euclid, $m_o(z)=w_o^\top z+b_o$, $m_n(z)=w_n^\top z+b_n$, và $w_o,w_n$ độc lập tuyến tính. Đặt $u=z-z_0$, $a=m_n(z_0)+\kappa>0$, $c=m_o(z_0)-\kappa\ge0$. Bài toán là:
$$\min_u\|u\|^2\ \ \text{s.t.}\ \ w_n^\top u\le -a,\quad w_o^\top u\ge -c.$$

> Bài toán khả thi và có nghiệm duy nhất $u^\*$:
> - **Trường hợp A:** nếu $a\,\dfrac{w_o^\top w_n}{\|w_n\|^2}\le c$ thì $u^\*=-\dfrac{a}{\|w_n\|^2}w_n$ và $\delta_\Delta=\dfrac{a}{\|w_n\|}$. Khi đó Δ-CE trùng với CE thông thường của $f_{new}$.
> - **Trường hợp B:** ngược lại, cả hai ràng buộc đều active và $u^\*=W^\top(WW^\top)^{-1}r$, với $W=\begin{bmatrix}w_n^\top\\ w_o^\top\end{bmatrix}$, $r=\begin{bmatrix}-a\\-c\end{bmatrix}$, và $\delta_\Delta=\sqrt{r^\top(WW^\top)^{-1}r}$.

*Chứng minh.* Hai pháp tuyến độc lập tuyến tính nên hệ $Wu=r$ có nghiệm, do đó bài toán khả thi. Hàm mục tiêu lồi chặt nên nghiệm duy nhất.

Ta dùng tính chất sau: nếu một ràng buộc không active tại $u^\*$ thì $u^\*$ cũng là nghiệm của bài toán bỏ ràng buộc đó (cực tiểu địa phương của bài toán lồi là cực tiểu toàn cục).
- Ràng buộc 1 luôn active. Nếu không, $u^\*$ sẽ là hình chiếu của $0$ lên nửa không gian 2, tức là $u^\*=0$ (vì $c\ge0$), nhưng $0$ vi phạm ràng buộc 1 (vì $a>0$). Mâu thuẫn.
- Nếu ràng buộc 2 không active thì $u^\*$ là hình chiếu lên nửa không gian 1, tức là $u_A$. Do đó: nếu $u_A$ thỏa ràng buộc 2 (đúng là điều kiện của trường hợp A) thì $u^\*=u_A$, vì $u_A$ tối ưu cho bài toán nới lỏng và khả thi cho bài toán gốc. Nếu không thì cả hai ràng buộc active, và $u^\*$ là nghiệm chuẩn nhỏ nhất của $Wu=r$. ∎

**Nhận xét (trường hợp song song).** Nếu $w_n=\lambda w_o$ với $\lambda>0$ thì bài toán khả thi khi và chỉ khi $a\le\lambda c$. Đặc biệt khi $f_{new}=f_{old}$ ($\lambda=1$, $b_n=b_o$) điều kiện trở thành $h(z_0)\le-2\kappa$, mâu thuẫn với $h\equiv0$ nếu $\kappa>0$. Điều này khớp với Bổ đề 1.

Luôn có $\delta_\Delta\ge a/\|w_n\|$ = khoảng cách CE thông thường của $f_{new}$.

### 3.5. Mệnh đề 4 (khôi phục hướng thay đổi, tuyến tính) — đã chứng minh, kèm cảnh báo

Đặt $\Delta w=w_n-w_o$ và $P=I-\dfrac{w_ow_o^\top}{\|w_o\|^2}$ (phép chiếu bỏ hướng mà mô hình cũ dùng).

> $Pu^\*\in\operatorname{span}\{P\Delta w\}$. Nói cách khác, sau khi bỏ thành phần theo $w_o$, Δ-CE di chuyển đúng theo phần thay đổi tham số vuông góc với mô hình cũ. Nếu $P\Delta w\neq0$ và $Pu^\*\neq0$ thì $|\cos(Pu^\*,P\Delta w)|=1$.

*Chứng minh.* Ở cả hai trường hợp, $u^\*\in\operatorname{span}\{w_n,w_o\}=\operatorname{span}\{w_o,\Delta w\}$. Áp dụng $P$, với $Pw_o=0$. ∎

> **Cảnh báo trung thực (cũng đã chứng minh).** Hiệu của hai CE chuẩn nhỏ nhất độc lập là $u_n-u_o=-\dfrac{a}{\|w_n\|^2}w_n+\dfrac{m_o(z_0)+\kappa}{\|w_o\|^2}w_o$, và nó **cũng** thỏa $P(u_n-u_o)\in\operatorname{span}\{P\Delta w\}$.
>
> Vậy trong trường hợp tuyến tính với CE tất định, **Δ-CE không hơn baseline B1 về hướng**. Lợi thế của Δ-CE phải đến từ ba nguồn khác:
> 1. tính witness (§3.1);
> 2. hành vi null (Mệnh đề 2);
> 3. thay đổi **phi cục bộ** và CE **không duy nhất** (§3.6).
>
> Giả thuyết H2–H3 được thiết kế đúng theo hướng này.

### 3.6. Những gì KHÔNG được chứng minh (giả thuyết, kiểm bằng thực nghiệm)

| Phát biểu | Trạng thái |
|---|---|
| Mệnh đề 4 chuyển sang deep net qua tuyến tính hóa cục bộ | **Heuristic.** Không có cận sai số; chỉ đúng khi $\delta_\Delta$ nhỏ so với bán kính tuyến tính |
| Khi thay đổi là *phi cục bộ* (xảy ra ở vùng xa $x$, ví dụ đảo nhãn một nhóm con), gradient-difference tại $x$ không thấy gì, còn Δ-CE tìm được | **Lập luận trực giác**, kiểm bằng điều kiện T3. Có ví dụ dựng tay: $h\equiv0$ quanh $x$ nhưng $h<-2\kappa$ ở vùng nhóm con, nên $\nabla h(x)=0$ trong khi $\mathcal F_\kappa\neq\emptyset$ |
| Khi bộ sinh CE không duy nhất (random restart, latent), CE differencing gây false attribution trên cặp mô hình null | **Ví dụ phản chứng tầm thường**: $f_o=f_n$ và tập CE có nhiều hơn một phần tử, nên hai lựa chọn khác nhau cho hiệu khác 0. Độ lớn hiệu ứng trên thực tế là câu hỏi thực nghiệm (H3) |
| Kết quả trong latent VAE phản ánh đúng thay đổi trên ảnh | Phụ thuộc decoder. Kiểm bằng oracle trên ảnh decode (G2) |

**Kiểm số trước khi viết (H4):** với Tier L (§4.1), nghiệm số của bộ giải Δ-CE phải khớp Mệnh đề 3 (sai số tương đối < $10^{-4}$) và Mệnh đề 4 ($|\cos| > 0.999$) trên 1000 cặp tuyến tính ngẫu nhiên.

---

## 4. Thiết kế thực nghiệm

### 4.1. Ba tầng dữ liệu

| Tier | Dữ liệu | Không gian tìm kiếm $\mathcal Z$ | Oracle factor | Chạy trên |
|---|---|---|---|---|
| **L** (sanity) | Tuyến tính ngẫu nhiên, $p\in\{5,20\}$ | $\mathbb R^p$ | Tham số đúng | CPU, vài giây |
| **0** (chính, manifold chính xác) | **Ellipses**: renderer tổng hợp khả vi 32×32 RGB | Không gian factor chuẩn hóa $[0,1]^5$; ảnh $=R(z)$ | Factor đúng tuyệt đối | CPU, có thể GPU |
| **1** (dữ liệu gần thực) | **Colored MorphoMNIST** 28×28 RGB | Latent VAE, $\dim=16$ | Oracle giải tích: màu nét, độ sáng nền, độ dày nét (đo bằng morphometry trên mask) | GPU nhỏ |

**Renderer Tier 0.** Pixel đối tượng là $\sigma\!\big(s\,(1-\tfrac{(u-c_x)^2}{A^2}-\tfrac{(v-c_y)^2}{B^2})\big)$, khả vi theo factor.

Năm factor:
- **tỉ lệ khung** $\rho=A/B$ (factor nhiệm vụ; nhãn $y=\mathbb 1[\rho>1]$, nhiễu nhãn 5%);
- **kích thước**;
- **hue đối tượng**;
- **độ sáng nền**;
- **vị trí ngang**.

Mọi factor được lấy mẫu độc lập và đều trên $[0,1]$ đã chuẩn hóa. Có thể tái dùng renderer/compositor của đề tài trước nếu phù hợp (đã pass 295/295).

**Tier 1.**
- Nhiệm vụ nhị phân: chữ số 4 vs 9.
- Tô màu nét bằng hue ngẫu nhiên; nền xám với độ sáng ngẫu nhiên; độ dày nét biến đổi bằng thickening/thinning kiểu Morpho-MNIST (Castro et al. 2019). Có thể tự cài bằng `skimage.morphology` nếu không cài được package.
- *Shape oracle* (nhận diện 4 vs 9) là CNN train trên ảnh xám đã chuẩn hóa độ dày. Oracle này được qualify tự động: yêu cầu accuracy ≥ 98% trên test thật. **Không cần nhãn người.**

### 4.2. Các thay đổi được tiêm (ground truth)

Quy trình chung: $f_{old}$ được train trên dữ liệu sạch (factor độc lập). $f_{new}$ là $f_{old}$ được fine-tune 3 epoch trên dữ liệu đã biến đổi như sau:

| Mã | Thay đổi | Cách tạo dữ liệu fine-tune | Factor ground truth $k^\*$ | Loại Δ-CE | Tier |
|---|---|---|---|---|---|
| **T1** | Thêm shortcut | hue tương quan với nhãn ($P(\text{hue}>0.5\mid y=1)=0.95$) | hue | N | 0, 1 |
| **T3** | Đảo nhãn nhóm con (phi cục bộ) | Tier 0: đảo nhãn khi kích thước > 0.75. Tier 1: đảo nhãn khi độ dày > ngưỡng q85 | kích thước / độ dày | N | 0, 1 |
| **T4** | Dịch ngưỡng factor nhiệm vụ | nhãn $=\mathbb 1[\rho>1.25]$ thay cho $\rho>1$ | tỉ lệ khung $\rho$ | N | 0 |
| **N0** | **Null** | Fine-tune trên dữ liệu sạch với seed khác (cùng phân phối) | — (không có thay đổi hệ thống) | N | 0, 1 |

Số cặp mô hình:
- T1, T3, T4: 3 seed mỗi loại.
- N0: **5 cặp seed** mỗi tier.
- Tổng: Tier 0 có 14 cặp, Tier 1 có 11 cặp.

**Ghi chú về N0.** Các mô hình khác seed có thể thực sự bất đồng ở một số vùng (hiệu ứng Rashomon). Witness tìm được trong N0 **không bị coi là sai**. Cái bị coi là sai là việc quy cho *một factor nhất quán* (xem metric SAR ở §4.5). Tỉ lệ witness trong N0 được báo cáo riêng như một phép đo Rashomon.

### 4.3. Phương pháp

**Δ-CE (đề xuất).** Với mỗi $z_0$, tối ưu bằng Adam (lr 0.05, 400 bước, $R=5$ random restart quanh $z_0$ với nhiễu $\sigma=0.05$):
$$
\mathcal L(z)=\underbrace{[m_n(z)+\kappa]_+ + [\kappa-m_o(z)]_+}_{\text{điều kiện witness}}+\lambda\,d(z_0,z)^2+\mu\,\mathcal R(z)
$$
- $\mathcal R$ = phạt ra khỏi miền $[0,1]^5$ (Tier 0), hoặc $\|z\|^2$ theo prior chuẩn (Tier 1).
- Lịch $\lambda$: bắt đầu 1.0, giảm ×0.5 nếu sau 200 bước chưa khả thi.
- Chọn nghiệm khả thi có $d$ nhỏ nhất trong các restart.
- Chỉ trả về witness nếu $d(z_0,z^\*)\le r$ (ngân sách), ngược lại trả về **"không tìm thấy"**.

**Siêu tham số khóa trước:**
- $\kappa$: logit-margin 0.5.
- $r$: bằng 2 × trung vị $\delta_{CE}$ thường của $f_{new}$ trên tập validation của từng tier, tính một lần trên cặp T1 seed 0 và dùng cố định.
- Mọi điều chỉnh sau khi khóa spec đều được ghi vào nhật ký quyết định.

**Baseline:**

| Mã | Mô tả | Ghi chú |
|---|---|---|
| **B1** | *CE differencing* (Artelt): CE thường của $f_{old}$ và của $f_{new}$, cùng bộ tối ưu, cùng $R$ restart, chọn độc lập; giải thích $=\phi(x'_{new})-\phi(x'_{old})$ | Baseline chính |
| **B1-det** | Như B1 nhưng tất định (một restart, cùng khởi tạo) | Tách hiệu ứng không duy nhất |
| **B3** | *Gradient difference*: $\nabla_z h(z_0)$. Tier 0: đọc trực tiếp trong không gian factor. Tier 1: bước dò độ dài $r/2$ theo $-\nabla_z h$, decode rồi đo bằng oracle | Baseline mạnh cho thay đổi cục bộ (Mệnh đề 4) |
| **B4** | *Feature-importance change* (theo hướng Muschalik): hiệu permutation importance theo factor giữa hai mô hình, ở mức toàn cục | Chỉ ở Tier 0 |

**Ablation:**
- **A1** bỏ proximity ($\lambda=0$, tương tự DiG-IN).
- **A2** bỏ ràng buộc $m_o\ge\kappa$ (= CE thường của $f_{new}$).
- **A3** $\kappa\in\{0, 0.5, 1.0\}$.

### 4.4. Đo factor

Vector thay đổi factor $\Delta\phi=\phi(D(z^\*))-\phi(D(z_0))$.
- Tier 0: $\phi$ là factor đúng.
- Tier 1: $\phi$ là oracle, chuẩn hóa theo độ lệch chuẩn đo trên tập dữ liệu thật.

### 4.5. Metric

| Metric | Định nghĩa | Vai trò |
|---|---|---|
| **Recovery@r** (chính) | $\frac1N\sum_i \mathbb 1[\text{có giải thích hợp lệ}]\cdot\mathbb 1[\arg\max_k|\Delta\phi_{ik}|=k^\*]$. Trường hợp không hợp lệ hoặc "không tìm thấy" tính là 0 | H1, H2 |
| Share | $|\Delta\phi_{k^\*}|/\sum_k|\Delta\phi_k|$, trung bình trên các giải thích hợp lệ | Phụ |
| Validity / found-rate | Tỉ lệ có witness trong ngân sách $r$ | Báo cáo |
| **SAR** (Systematic Attribution Rate) | Trên các cặp N0: tỉ lệ cặp có $\max_k \overline{\text{share}}_k\ge0.5$ (trung bình trên $N$ điểm) | H3 |
| Rashomon rate | Found-rate của Δ-CE trên N0 | Báo cáo |
| Plausibility (Tier 1) | Khoảng cách kNN trong không gian feature của shape oracle tới dữ liệu thật; tỉ lệ ảnh mà shape oracle tin cậy > 0.9 | Báo cáo |
| Runtime | Giây/mẫu | Báo cáo |

Số mẫu: $N=300$ điểm mỗi cặp ở Tier 0, $N=200$ ở Tier 1, lấy ngẫu nhiên trong các điểm mà cả hai mô hình cùng dự đoán đúng với margin $\ge\kappa$.

---

## 5. Giả thuyết, kiểm định và tiêu chí quyết định (khai báo trước)

### 5.1. Giả thuyết

- **H4 (sanity lý thuyết, Tier L).** Bộ giải khớp Mệnh đề 3–4 với dung sai ở §3.6. *Thất bại ở đây nghĩa là có bug; phải sửa trước khi tiếp tục.*
- **H1 (Δ-CE vs B1, gộp T1 và T3, Tier 1).** Recovery@r(Δ-CE) − Recovery@r(B1) > 0.
- **H2 (thay đổi phi cục bộ, T3, Tier 0 và 1).** Recovery@r(Δ-CE) − Recovery@r(B3) > 0.
- **H3 (null, N0).** SAR(Δ-CE) = 0/5 ở mỗi tier. SAR(B1) được báo cáo; kỳ vọng > 0 với B1 không tất định.

### 5.2. Kiểm định

- Paired bootstrap (10 000 lần) trên các điểm, phân tầng theo cặp mô hình. CI 95%.
- Hiệu chỉnh Holm cho ba kiểm định chính H1, H2-Tier0, H2-Tier1.
- Báo cáo kết quả từng seed; không chỉ báo cáo trung bình.

### 5.3. Kết quả được dự đoán trước là hòa (không tính là thất bại)

Theo Mệnh đề 4, ở các thay đổi cục bộ T1 và T4, Δ-CE ≈ B3 ≈ B1-det. Dự đoán: |chênh lệch Recovery| ≤ 0.10. **Nếu kết quả đúng như vậy thì đó là xác nhận lý thuyết, không phải thất bại.**

### 5.4. Route

| Route | Điều kiện | Câu chuyện paper |
|---|---|---|
| **A** (phương pháp + lý thuyết) | H4, H3 đạt; ít nhất một trong H1/H2 có CI loại trừ 0 sau Holm | "Δ-CE: witness tối thiểu cho model change, có hành vi null và bắt được thay đổi phi cục bộ" |
| **A-strong** | Như A, thêm hiệu ứng ≥ 0.15 | Như A, claim mạnh hơn |
| **B** (lý thuyết + đo lường) | H4 đạt; H1 và H2 đều không đạt | "Khi nào so sánh counterfactual giải thích được model change?" Gồm Bổ đề 1, Mệnh đề 2–4, và bằng chứng thực nghiệm rằng baseline đơn giản đủ dùng với thay đổi cục bộ. Vẫn nộp được, phù hợp với special session về reliable counterfactuals |
| **Dừng** | G0 không đạt được sau 2 lần sửa thiết kế (đến 21/10) | Báo cáo lại, chọn lại đề tài |

---

## 6. Gate (theo thứ tự rẻ nhất trước)

| Gate | Ngày | Kiểm tra | Tiêu chí đạt | Nếu không đạt |
|---|---|---|---|---|
| **G-lit** | 17/10 | Novelty full text (§2.3) | Không có công trình trùng định nghĩa và benchmark | Đổi trọng tâm sang benchmark + chứng chỉ null; báo cáo lại |
| **G0** (hiện tượng) | 20/10 | Tier 0: các thay đổi tiêm vào có thật sự làm thay đổi mô hình không. Đo độ nhạy can thiệp $S_k(f)=P(\text{dự đoán đổi}\mid\text{lấy mẫu lại factor }k)$ trên test | Với T1/T3/T4: $S_{k^\*}(f_{new})-S_{k^\*}(f_{old})\ge0.20$. Với N0: $\le0.05$ với mọi $k$. Cả hai mô hình có accuracy ≥ 95% trên phân phối sạch, trừ vùng bị tiêm | Tăng mức tương quan hoặc số epoch fine-tune (tối đa 2 lần), rồi Dừng |
| **G1** | 21/10 | Tier L (H4) và Tier 0 pilot với 1 seed, $N=100$ | H4 đạt; found-rate Δ-CE ≥ 0.6 trên T1; thời gian chạy đo thật | Sửa bộ giải (restart, lịch $\lambda$) |
| **G2** (oracle Tier 1) | 24/10 | Oracle trên ảnh VAE tái tạo so với factor đúng | Spearman ≥ 0.9 (hue, nền), ≥ 0.8 (độ dày); shape oracle accuracy ≥ 98% | Bỏ T3-độ dày, thay bằng T3-độ sáng nền (oracle dễ hơn) |
| **G3** | 1/11 | Đủ seed, chạy kiểm định | — | Chọn Route A hoặc B |

---

## 7. Ngân sách tính toán (ước tính, sẽ đo thật ở G1)

| Hạng mục | Ước tính |
|---|---|
| Tier L + Tier 0 toàn bộ (14 cặp × train/fine-tune CNN ~1 phút, Δ-CE + baseline 300 điểm × 5 restart × 400 bước) | ≤ 1 GPU-giờ, hoặc vài giờ CPU |
| Tier 1: VAE 30 epoch (~15 phút), shape oracle (~3 phút), 11 cặp classifier (~2 phút mỗi cặp), Δ-CE + baseline 200 điểm | ~2–3 GPU-giờ |
| Ablation A1–A3 (Tier 0 đủ, Tier 1 một seed) | ~1 GPU-giờ |
| Dự phòng chạy lại | ~2 GPU-giờ |
| **Tổng** | **~6 GPU-giờ (trần 12)** |

**Quy tắc vận hành:**
- Mỗi job GPU < 60 phút và có checkpoint.
- Gộp tất cả cặp của một tier vào ít job.
- Tier 0 chạy local để không tốn lượt submit.
- Lưu nhật ký thời gian thật để cập nhật ước tính ở G1.

---

## 8. Timeline (hôm nay: thứ Năm 15/10/2026)

| Giai đoạn | Ngày | Việc | Mốc |
|---|---|---|---|
| W0 | 15–17/10 | Đọc full text các công trình gần nhất; khóa spec; code renderer Tier 0 và Tier L | **G-lit 17/10**, khóa spec |
| W1 | 18–21/10 | Train/fine-tune Tier 0, đo độ nhạy can thiệp; bộ giải Δ-CE và baseline; H4 | **G0 20/10, G1 21/10** |
| W2 | 22–27/10 | Tier 1: dữ liệu, VAE, oracle, shape oracle; pilot | **G2 24/10** |
| W3 | 28/10–1/11 | Chạy đủ seed hai tier, ablation, bootstrap | **G3 1/11**: chọn route |
| W4 | 2–8/11 | Viết bản nháp 6 trang (kiểm lại page limit trong author guidelines của ESANN), hình, bảng | Bản nháp 8/11 |
| W5 | 9–12/11 | Kiểm lại chứng minh, review nội bộ, chạy lại để tái lập | — |
| W6 | 13–16/11 | Chỉnh sửa, chuẩn bị bản nộp | **Nộp 16/11** |
| Dự phòng | 17–18/11 | Không lên kế hoạch việc gì | Hạn cứng 18/11 |

---

## 9. Dàn ý paper (khoảng 6 trang)

1. **Introduction** (0.75 trang): model updates, negative flips, khoảng trống giữa "so sánh CE" và witness.
2. **Related work** (0.5 trang): explaining model change (Artelt, Hinder, Muschalik), robust CE under model change (đối ngẫu), DiG-IN, ECCCo.
3. **Differential counterfactuals** (1.25 trang): Định nghĩa 1, Bổ đề 1, Mệnh đề 2–4, cảnh báo ở §3.5. Chứng minh ngắn đưa vào chú thích hoặc in gọn.
4. **Benchmark with injected changes** (1 trang): các tier, bảng thay đổi tiêm vào, metric.
5. **Experiments** (1.75 trang): một bảng chính (Recovery@r theo phương pháp × thay đổi × tier), một hình (witness Tier 0 trong không gian factor và ví dụ Tier 1), kết quả null/SAR.
6. **Discussion & limitations** (0.5 trang): chứng chỉ không chặt cho deep net, phụ thuộc decoder, chỉ có hai tier tổng hợp.

---

## 10. Rủi ro

| Rủi ro | Xác suất | Ứng phó |
|---|---|---|
| Phát hiện công trình trùng ở G-lit | Thấp–TB | Chuyển trọng tâm sang benchmark + chứng chỉ null; trích dẫn và so sánh trực tiếp |
| B3 bằng Δ-CE ở cả T3 (mô hình nội suy mượt nên thay đổi "lan" về gần $x$) | TB | Đây là kết quả khoa học hợp lệ, đi theo Route B. Thử thêm nhóm con xa hơn (kích thước > 0.85) **chỉ khi đã khai báo trước** như một biến thể |
| Bộ giải không tìm được witness (bài toán không lồi) | TB | Restart, lịch $\lambda$, khởi tạo từ CE thường của $f_{new}$. Báo cáo found-rate trung thực |
| VAE quá mờ nên oracle độ dày nhiễu | TB | Phương án ở G2: thay bằng độ sáng nền |
| Hạ tầng hoặc quota | Thấp | Tier 0 chạy local hoàn toàn; Tier 1 tổng cộng < 4 job |
| Giới hạn số trang | TB | Ưu tiên Tier 0 + lý thuyết; Tier 1 rút gọn còn một bảng |

---

## 11. Nhật ký giả định cần xác nhận

- Page limit và template của ESANN 2027 (kiểm trên esann.org, mục *Author guidelines*).
- ESANN 2027 có special session về counterfactual/XAI hay không (kiểm mục *Special sessions*).
- Có cài được Morpho-MNIST trên nền tảng chạy không (nếu không thì tự cài phép đo bằng `skimage`).
- Tài khoản và quota GPU dùng cho đề tài này.

---

## 12. Tài liệu tham khảo chính

1. Artelt, A., Hinder, F., Vaquet, V., Feldhans, R., Hammer, B. *Contrastive Explanations for Explaining Model Adaptations.* arXiv:2104.02459, 2021.
2. Artelt, A. et al. *Contrasting Explanations for Understanding and Regularizing Model Adaptations.* Neural Processing Letters, 2022. https://link.springer.com/article/10.1007/s11063-022-10826-5
3. Hinder, F., Hammer, B. *Counterfactual Explanations of Concept Drift.* arXiv:2006.12822, 2020.
4. Hammer, B., Hüllermeier, E. *Interpretable Machine Learning: On the Problem of Explaining Model Change.* 2021.
5. Muschalik, M., Fumagalli, F., Hammer, B., Hüllermeier, E. *Agnostic Explanation of Model Change based on Feature Importance.* KI – Künstliche Intelligenz, 2022.
6. *Explaining Concept Drift through the Evolution of Group Counterfactuals.* arXiv:2509.09616, 2025.
7. Augustin, M. et al. *DiG-IN: Diffusion Guidance for Investigating Networks.* CVPR 2024. arXiv:2311.17833.
8. Weng, N. et al. *Fast Diffusion-Based Counterfactuals for Shortcut Removal and Generation (FastDiME).* ECCV 2024. arXiv:2312.14223.
9. Altmeyer, P. et al. *Faithful Model Explanations through Energy-Constrained Conformal Counterfactuals (ECCCo).* AAAI 2024.
10. Hamman, F. et al. *Robust Counterfactual Explanations for Neural Networks With Probabilistic Guarantees.* ICML 2023.
11. *Provably Robust Bayesian Counterfactual Explanations under Model Changes (PSCE).* arXiv:2601.16659, 2026.
12. *Probabilistically robust counterfactual explanations under model changes.* Artificial Intelligence, 2025.
13. Pawelczyk, M., Broelemann, K., Kasneci, G. *On Counterfactual Explanations under Predictive Multiplicity.* UAI 2020.
14. *A Geometric View of Counterfactual Behavior: Interaction of Boundary Proximity and Local Support.* arXiv:2606.04209, 2026.
15. Kaden, M., Paassen, B., Hammer, B., Schubert, R., Villmann, T. *Reliable Counterfactuals for Machine Learning Models — Current Aspects and Perspectives.* ESANN 2026.
16. Castro, D. C. et al. *Morpho-MNIST: Quantitative Assessment and Diagnostics for Representation Learning.* JMLR 2019.
17. Wachter, S., Mittelstadt, B., Russell, C. *Counterfactual Explanations without Opening the Black Box.* Harvard JL & Tech, 2017.
18. Yan, S. et al. *Positive-Congruent Training: Towards Regression-Free Model Updates.* CVPR 2021. *(động lực negative flips; chưa tra lại trong phiên này)*
