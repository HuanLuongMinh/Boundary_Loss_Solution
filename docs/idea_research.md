# RESEARCH PROPOSAL & EXPERIMENTAL PROTOCOL

**Tên đề tài dự kiến:** Dynamic Boundary-Aware Loss via Adaptive BCE-Affinity Coupling for Fine-Grained Semantic Segmentation in Remote Sensing Imagery  
**Lĩnh vực:** Computer Vision / Remote Sensing (Phân đoạn ngữ nghĩa ảnh vệ tinh)  
**Mục tiêu chính:** Nâng cao độ chính xác phân đoạn tại các ranh giới đối tượng tiếp giáp, chồng lấn ở quy mô sub-pixel và pixel siêu nhỏ.

---

## 1. Đặt Vấn Đề & Động Lực Nghiên Cứu (Motivation)

Trong phân đoạn ngữ nghĩa ảnh vệ tinh và viễn thám:
* **Nhiễu ranh giới (Boundary Ambiguity):** Sự chồng lấn và tiếp giáp giữa các lớp địa vật (như ranh giới thửa đất, viền công trình xây dựng, đường giao thông, tán rừng) thường chỉ diễn ra trên dải $1 - 3$ pixel.
* **Hạn chế của Region Loss:** Các hàm mất mát vùng truyền thống (Cross-Entropy, Focal, Dice Loss) bị chi phối bởi các pixel phần ruột (interior pixels) chiếm đa số, dẫn đến hiện tượng trơn hóa (over-smoothing) và mất chi tiết góc cạnh ranh giới.
* **Hạn chế của Boundary Loss thông thường:** Khi kết hợp BCE cạnh và Affinity Loss với hệ số tĩnh, mô hình dễ bị mất cân bằng gradient:
  * Giai đoạn đầu: Mạng chưa định vị tốt đối tượng, việc áp đặt tương quan affinity sớm dễ dẫn đến lan truyền nhiễu.
  * Giai đoạn sau: Cần siết chặt tính nhất quán giữa các pixel lân cận cùng lớp và tách biệt các pixel khác lớp ở viền.

---

## 2. Phương Pháp Đề Xuất (Proposed Method)

### 2.1. Cấu trúc Hàm Mất Mát Toàn Cục
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{region}} + \alpha \cdot \mathcal{L}_{\text{boundary\_dynamic}}$$

* $\mathcal{L}_{\text{region}}$: Hàm mất mát vùng (Cross-Entropy + Dice/Focal Loss).
* $\alpha$: Siêu tham số điều phối mức độ đóng góp của thành phần biên.
* $\mathcal{L}_{\text{boundary\_dynamic}}$: Hàm mất mát ranh giới thích ứng động.

### 2.2. Cơ chế Thích Ứng Động (Dynamic BCE-Affinity Coupling)
$$\mathcal{L}_{\text{boundary\_dynamic}} = \lambda_1(t) \cdot \mathcal{L}_{\text{BCE\_edge}} + \lambda_2(t) \cdot \mathcal{L}_{\text{Affinity}}$$

* **$\mathcal{L}_{\text{BCE\_edge}}$ (Edge Localization):** Đóng vai trò định vị nhị phân pixel ranh giới (Edge vs. Non-edge), trích xuất thông qua toán tử Sobel/Canny từ ground-truth mask.
* **$\mathcal{L}_{\text{Affinity}}$ (Local Semantic Consistency):** Đo lường sự tương đồng đặc trưng (cosine similarity / L2 distance) giữa các cặp pixel lân cận trong cửa sổ cục bộ $K \times K$ tại vùng ranh giới:
  $$\mathcal{L}_{\text{Affinity}} = \frac{1}{|\mathcal{P}|} \sum_{(i, j) \in \mathcal{P}} \left[ A_{ij} \cdot D(f_i, f_j) + (1 - A_{ij}) \cdot \max(0, m - D(f_i, f_j)) \right]$$
  *(với $A_{ij} \in \{0, 1\}$ chỉ ra hai pixel $i, j$ có cùng nhãn ngữ nghĩa hay không; $f_i, f_j$ là embedding feature; $m$ là margin).*
* **Hệ số động $\lambda_1(t), \lambda_2(t)$:** Điều chỉnh trọng số tự động theo thời gian huấn luyện $t$ (Curriculum Learning) hoặc dựa trên độ bất định / độ lớn gradient (Gradient Magnitude Balancing):
  * Ban đầu: $\lambda_1(t)$ chiếm ưu thế để học định vị thô đường biên.
  * Về sau: $\lambda_2(t)$ tăng dần để tinh chỉnh sự phân tách giữa các lớp sát viền.

---

## 3. Chiến Lược Thực Nghiệm 2 Giai Đoạn (Experimental Design)

Quy trình kết hợp **Fixed Split (Ablation & Tuning)** và **5-Fold Cross-Validation (Main Benchmark)** nhằm tối ưu chi phí tính toán nhưng vẫn đảm bảo tính khách quan học thuật cao nhất.

[ TOÀN BỘ DATASET GEO-TIFF ]
                            │
   ┌────────────────────────┴────────────────────────┐
   ▼                                                 ▼
[ GIAI ĐOẠN 1: FIXED SPLIT ]             [ GIAI ĐOẠN 2: 5-FOLD CV ]
• Cố định 80% Train / 20% Val            • 5 Folds phân vùng không gian độc lập
• Quét siêu tham số α (4 runs)           • Huấn luyện Baseline (5 runs)
• Ablation Study 4 biến thể (4 runs)     • Huấn luyện Proposed Method (5 runs)
==> Tổng: 8 runs                         ==> Tổng: 10 runs

---

### Giai đoạn 1: Tuning Siêu tham số & Bóc tách Thành phần (Fixed Split)
* **Môi trường:** Cố định phân chia 80% Train / 20% Validation (`seed=42`).
* **Mục tiêu:**
  1. Quét tìm $\alpha^* \in \{0.1, 0.5, 1.0, 2.0\}$.
  2. Bóc tách vai trò của từng thành phần loss (Ablation Study).

#### Danh mục thực nghiệm Pha 1:
| Run | Cấu hình | Hàm Loss chi tiết | Trọng số áp dụng |
| :---: | :--- | :--- | :--- |
| 1 | **Baseline** | $\mathcal{L}_{\text{region}}$ | $\alpha = 0$ |
| 2 | **+ BCE Edge** | $\mathcal{L}_{\text{region}} + \alpha \mathcal{L}_{\text{BCE\_edge}}$ | $\alpha = \alpha^*$ |
| 3 | **+ Static Boundary** | $\mathcal{L}_{\text{region}} + \alpha (\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Affinity}})$ | $\lambda_1 = 1.0, \lambda_2 = 1.0, \alpha = \alpha^*$ |
| 4 | **+ Dynamic Boundary (Ours)**| $\mathcal{L}_{\text{region}} + \alpha \mathcal{L}_{\text{boundary\_dynamic}}$ | Động $\lambda_1(t), \lambda_2(t), \alpha = \alpha^*$ |

*(Chi phí Pha 1: $4 \text{ runs quét } \alpha + 4 \text{ runs ablation} = 8\text{ runs}$).*

---

### Giai đoạn 2: Kiểm Định Chéo Khái Quát Hóa (5-Fold Cross-Validation)
* **Môi trường:** 5-Fold Spatial Cross-Validation (phân chia theo khối địa lý hoặc khu vực độc lập để chống tự tương quan không gian).
* **Mục tiêu:** Kiểm chứng tính bền vững trước nhiễu nhãn sub-pixel và sự phân bố lớp mất cân bằng qua nhiều vùng địa hình khác nhau.

#### Danh mục thực nghiệm Pha 2:
1. **Chạy 5 Folds cho Baseline:** $\mathcal{L}_{\text{region}}$ trên 5 fold $\rightarrow$ Thu được $\text{Mean} \pm \text{Std}$.
2. **Chạy 5 Folds cho Proposed:** $\mathcal{L}_{\text{total}}$ với cơ chế động và $\alpha^*$ tối ưu trên 5 fold $\rightarrow$ Thu được $\text{Mean} \pm \text{Std}$.

*(Chi phí Pha 2: $5 + 5 = 10\text{ runs}$).*

---

## 4. Hệ Thống Metric Đánh Giá Chuyên Biệt

1. **Chỉ số Vùng (Region Metrics):**
   * $\text{mIoU}$ (Mean Intersection over Union)
   * $\text{F1-Score / Dice per Class}$
   * $\text{Overall Accuracy (OA)}$
2. **Chỉ số Ranh giới (Boundary-Specific Metrics):**
   * $\text{Boundary IoU (Trimap-based)}$ tại các bán kính $d \in \{1, 3, 5\text{ pixels}\}$.
   * $\text{Boundary F1-Score (BF-Score)}$ đánh giá độ khớp đường bao contour.
   * $\text{Average Surface Distance (ASD)}$ đo khoảng cách sai lệch bề mặt trung bình.

---

## 5. Mẫu Bảng Biểu Trình Bày Trong Bài Báo Khoa Học

### Bảng 1: Kết quả Ablation Study (Đánh giá trên Fixed Validation Split)
| Phương pháp | $\mathcal{L}_{\text{BCE}}$ | $\mathcal{L}_{\text{Affinity}}$ | Cơ chế trọng số | mIoU (%) | Boundary IoU ($d=3$) | BF-Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline ($\mathcal{L}_{\text{region}}$) | — | — | — | 74.20 | 61.30 | 65.40 |
| Variant 1 | ✓ | — | Cố định ($\alpha^*$) | 74.85 | 63.10 | 67.20 |
| Variant 2 | ✓ | ✓ | Tĩnh ($\lambda_1=\lambda_2=1$) | 75.10 | 64.05 | 68.15 |
| **Proposed Method** | ✓ | ✓ | **Động $\lambda_1(t), \lambda_2(t)$** | **76.05** | **66.40** | **70.50** |

---

### Bảng 2: Kết quả Thử nghiệm Chính (Main Benchmark via 5-Fold Spatial CV)
| Phương pháp | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean $\pm$ Std** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Model** | 74.12 | 73.85 | 74.50 | 73.90 | 74.23 | $74.12 \pm 0.25$ |
| **Proposed Method** | **76.25** | **75.90** | **76.45** | **76.02** | **76.38** | $\mathbf{76.20 \pm 0.22}$ |
| **Cải thiện ($\Delta$)** | **+2.13** | **+2.05** | **+1.95** | **+2.12** | **+2.15** | $\mathbf{+2.08}$ |
