# 🧠 Machine Learning: From Foundations to Practice

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0%2B-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Kho lưu trữ toàn diện về các thuật toán Machine Learning kinh điển:** Kết hợp hài hòa giữa **toán học nền tảng (cài đặt từ đầu - hand-coded from scratch)** và **ứng dụng thực chiến với Scikit-learn**.

---

## 📌 Giới thiệu dự án

Dự án này được xây dựng nhằm cung cấp lộ trình học tập, nghiên cứu và thực hành Machine Learning bài bản từ bản chất toán học đến giải quyết bài toán thực tế:
- **Hiểu sâu bản chất:** Tự lập trình (hand-coded) các độ đo, hàm mục tiêu và cơ chế hoạt động của thuật toán (như Gini Impurity, khoảng cách khoảng không gian, trọng số phân nhánh, ma trận tương quan).
- **Trực quan hóa sinh động:** Minh họa trực quan không gian đặc trưng, biên quyết định (decision boundary), tác động của chuẩn hóa dữ liệu (Feature Scaling).
- **Ứng dụng thực tế (Real-world Projects):** Xử lý bài toán thực tiễn từ phát hiện gian lận tài chính (*Credit Card Fraud Detection*), phân loại văn bản (*NLP / Bag-of-Words*), đến dự báo kinh tế & doanh thu.

---

## 📂 Cấu trúc dự án

```text
Machine Learning/
│
├── 📁 Linear-Regression/             # Hồi quy tuyến tính & Bài toán dự báo liên tục
│   ├── linear-regression-aivn.ipynb # Thực hành Linear Regression (OLS & Gradient Descent)
│   ├── [Description]-...pdf         # Tài liệu giải thích lý thuyết toán học
│   └── data/
│       ├── advertising.csv          # Dữ liệu ngân sách quảng cáo & Doanh thu
│       └── BTC-Daily.csv            # Dữ liệu giá Bitcoin theo ngày
│
├── 📁 K-Nearest-Neighbor/           # Thuật toán láng giềng gần nhất (KNN)
│   ├── KNN_Classification/          # KNN cho bài toán phân loại
│   │   ├── 1.KNN-IRIS-1D.ipynb      # KNN trực quan trên 1 chiều không gian
│   │   ├── 2.KNN-IRIS-2D.ipynb      # KNN trực quan trên 2 chiều không gian
│   │   ├── 3.KNN-IRIS-2D - WhyScaling.ipynb   # Thực nghiệm: Tại sao cần Feature Scaling?
│   │   ├── 4.KNN-IRIS-2D - UsingScaling.ipynb # Áp dụng MinMaxScaler / StandardScaler
│   │   ├── 5.BoW_KNN.ipynb          # Xử lý ngôn ngữ tự nhiên: Bag-of-Words + KNN
│   │   ├── 6.KNN_text_cls.ipynb     # Phân loại văn bản hoàn chỉnh
│   │   └── *.csv                    # Dữ liệu thực nghiệm Iris thu gọn
│   ├── KNN_Regression/              # KNN cho bài toán hồi quy
│   │   ├── Salary - simple.ipynb    # Dự báo mức lương theo số năm kinh nghiệm
│   │   └── Salary_Data_simple.csv
│   ├── votes_skills_anon.ipynb      # Dự án phân tích & phân cụm kỹ năng người dùng
│   └── votes_skills_anon.csv
│
├── 📁 Decision-Tree/                # Cây quyết định (Decision Tree - CART)
│   ├── Decision_Tree_Warmup_1.ipynb # Cài đặt công thức Gini Impurity & Gini Split từ đầu
│   ├── Decision_Tree_Warmup_2.ipynb # Xây dựng cây quyết định & trực quan hóa với Scikit-learn
│   └── Decision_Tree_Credit_Card_Fraud_Detection.ipynb # Project: Phát hiện giao dịch gian lận thẻ tín dụng
│
├── .gitignore                       # Cấu hình loại bỏ cache, venv và tập dữ liệu lớn
└── README.md                        # Tài liệu hướng dẫn dự án
```

---

## 🎯 Chi tiết các chuyên đề & Thuật toán

### 1. 📈 Linear Regression (Hồi quy tuyến tính)
* **Toán học cốt lõi:** Phương pháp bình phương tối thiểu (Ordinary Least Squares - OLS), hàm mất mát MSE/RMSE, thuật toán Gradient Descent.
* **Nội dung thực hành:**
  * Xử lý dữ liệu đơn biến (Simple) và đa biến (Multiple Linear Regression).
  * Phân tích tác động của ngân sách tiếp thị (TV, Radio, Newspaper) lên doanh số (`advertising.csv`).
  * Khảo sát chuỗi dữ liệu tài chính lịch sử (`BTC-Daily.csv`).

---

### 2. 📍 K-Nearest Neighbors (K-Láng giềng gần nhất)
* **Toán học cốt lõi:** Khoảng cách Euclidean, Manhattan; cơ chế voting đa số và trọng số khoảng cách.
* **Điểm nổi bật về mặt kỹ thuật:**
  * **Trực quan hóa không gian chiều:** Xem xét sự biến đổi của biên quyết định khi tăng chiều từ 1D lên 2D.
  * **Chuyên đề chuyên sâu về Feature Scaling:** Thử nghiệm đối chiếu chứng minh KNN bị sai lệch nghiêm trọng nếu không chuẩn hóa các đặc trưng có đơn vị đo khác nhau (MinMaxScaler vs. StandardScaler).
  * **Ứng dụng NLP:** Biến đổi văn bản thành vector bằng **Bag-of-Words (BoW)** và phân loại cảm xúc/chủ đề văn bản với KNN.
  * **KNN Regression:** Ứng dụng dự báo giá trị liên tục trên tập dữ liệu lương.

---

### 3. 🌳 Decision Trees (Cây quyết định - CART)
* **Toán học cốt lõi:**
  * Độ vẩn đục Gini: $\text{Gini}(D) = 1 - \sum_{i=1}^C p_i^2$
  * Độ lợi phân chia có trọng số: $\text{Gini}_A(D) = \sum_{k} \frac{|D_k|}{|D|} \text{Gini}(D_k)$
  * Tiêu chuẩn chọn thuộc tính tối ưu để tạo nút rẽ nhánh (Best Split Criteria).
* **Nội dung thực hành:**
  * **Hand-coded Step-by-Step:** Tự viết hàm Python tính toán toàn bộ chỉ số Gini, Gini Gain trên dữ liệu phân loại mà không dùng thư viện ngoài.
  * **Scikit-learn Integration:** Huấn luyện `DecisionTreeClassifier`, xuất luật `export_text` và vẽ cây quyết định trực quan (`plot_tree`).
  * **Dự án thực tế - Credit Card Fraud Detection:** 
    * Xử lý tập dữ liệu mất cân bằng nghiêm trọng (Imbalanced Data).
    * Tinh chỉnh siêu tham số (`max_depth`, `min_samples_split`, `criterion`) để hạn chế Overfitting.
    * Đánh giá mô hình đa chiều qua Precision, Recall, F1-Score và đường cong ROC-AUC.

---

## 🛠️ Công nghệ & Thư viện sử dụng

| Thư viện | Phiên bản khuyến nghị | Mục đích sử dụng |
| :--- | :--- | :--- |
| **Python** | `3.10+` | Ngôn ngữ phát triển chính |
| **NumPy** | `2.0+` | Xử lý mảng đa chiều, tính toán đại số tuyến tính & véc-tơ hóa |
| **Pandas** | `2.0+` | Thao tác, làm sạch, mã hóa (One-Hot) và phân tích dữ liệu bảng |
| **Scikit-learn** | `1.4+` | Xây dựng pipeline, huấn luyện mô hình ML và đánh giá chất lượng |
| **Matplotlib** | `3.8+` | Vẽ đồ thị hàm số, biểu đồ phân tán và trực quan hóa cây |
| **Seaborn** | `0.13+` | Trực quan hóa ma trận nhầm lẫn (Confusion Matrix) & phân phối dữ liệu |

---

## 🚀 Hướng dẫn cài đặt & Chạy dự án

### 1. Clone repository về máy

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "Machine Learning"
```

### 2. Khởi tạo môi trường ảo (Virtual Environment)

* **Trên Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

* **Trên macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài đặt các thư viện phụ thuộc

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 4. Khởi chạy Jupyter Notebook

```bash
jupyter notebook
```
> Trình duyệt sẽ tự động mở, bạn có thể truy cập vào các thư mục `Linear-Regression`, `K-Nearest-Neighbor` hoặc `Decision-Tree` để bắt đầu chạy các notebook.

---

## 📊 Dữ liệu (Datasets)

* Các tệp dữ liệu kích thước nhỏ và vừa (`iris_*.csv`, `advertising.csv`, `Salary_Data_simple.csv`, `votes_skills_anon.csv`) đã được đính kèm sẵn trong từng thư mục tương ứng.
* Đối với tập dữ liệu lớn **Credit Card Fraud Detection** (`creditcard.csv` ~150MB, đã được cấu hình trong `.gitignore` theo chính sách của GitHub), bạn có thể tải trực tiếp từ [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) và đặt vào thư mục `Decision-Tree/`.

---

## 📜 Giấy phép (License)

Dự án được phân phối dưới giấy phép **MIT License**. Bạn hoàn toàn có thể tự do sử dụng, chỉnh sửa và chia sẻ cho mục đích học tập và nghiên cứu cá nhân.
