# Otto Recommender System

Dự án này là một phần của bài tập lớn môn học, nhằm mục đích xây dựng và tối ưu hóa một hệ thống gợi ý (Recommender System) có khả năng dự đoán nhiều hành vi người dùng (`clicks`, `carts`, `orders`) dựa trên dữ liệu từ cuộc thi [OTTO - Multi-Objective Recommender System trên Kaggle](https://www.kaggle.com/competitions/otto-recommender-system).

## 1. Cài đặt (Installation)

Dự án được phát triển chủ yếu trên môi trường Kaggle Notebook để tận dụng GPU và các thư viện cài đặt sẵn. Để chạy lại trên môi trường local, bạn cần đảm bảo các thư viện sau đã được cài đặt.

**Yêu cầu:**
*   Python 3.11+
*   PyTorch (phiên bản tương thích CUDA nếu có GPU)
*   Các thư viện phụ trợ khác

**Cách cài đặt:**

1.  Clone repo này về máy:
    ```bash
    git clone https://github.com/triet4p/otto-rec-sys.git
    cd otto-rec-sys
    ```

2.  Cài đặt các thư viện cần thiết. 
    ```bash
    pip install requirements.txt
    ```

    Hoặc dùng uv
    ```bash
    pip install uv
    uv sync
    ```

3.  **Dữ liệu:**
    *   Tải bộ dữ liệu từ [OTTO Kaggle Competition](https://www.kaggle.com/competitions/otto-recommender-system/data).
    *   Để chạy các notebook Local CV, bạn cần tải bộ dữ liệu validation đã được xử lý sẵn tại [OTTO - Validation](https://www.kaggle.com/datasets/radek1/otto-validation).
    *   Các ma trận co-visitation đã được tính toán trước có thể được tải về từ [đây](URL_TO_YOUR_KAGGLE_DATASET).

## 2. Cấu trúc Thư mục (Code Structure)

Dự án được tổ chức theo cấu trúc module hóa để dễ dàng quản lý, bảo trì và mở rộng.

```
otto-rec-sys/
│
├── data/                     # (Thư mục tùy chọn) Nơi lưu các file dữ liệu nhỏ
│
├── notebooks/                # Chứa các notebook thực thi chính
│
├── src/                      # Chứa toàn bộ source code được module hóa
│   ├── core/
│   │   ├── config.py         # Chứa các hằng số, cấu hình, đường dẫn
│   │   └── utils.py          # Các hàm tiện ích chung
│   │
│   ├── candidate_generation/
│   │   ├── covisit.py        # Logic tạo các ma trận co-visitation
│   │   ├── gnn.py            # Logic tạo ứng viên từ GNN/Embedding
│   │   └── loader.py         # Các hàm tải ma trận
│   │
│   ├── pipeline/
│   │   ├── preprocess.py     # Các hàm tiền xử lý, tạo ứng viên
│   │   ├── training.py       # Các hàm tạo training set, feature engineering
│   │   └── validation.py     # Hàm tính điểm Local CV
│
└── README.md                 # File hướng dẫn này
```

## 3. Quy trình Thực thi (How to Run)

Quy trình hoàn chỉnh bao gồm 3 bước chính, tương ứng với các notebook.

### Bước 1: Tạo các ma trận Covisit(Pre-computation)
*   **Mục đích:** Tính toán trước các ma trận Co-visitation và các file Embedding tốn nhiều tài nguyên. Bước này chỉ cần chạy **một lần**.
*   **Notebook:** [`Covisit-generation`](./notebooks/otto-recsys-covisit-matrix-generation.ipynb)
*   **Đầu ra:** Một Kaggle Dataset chứa các file `.pqt` của ma trận.

Với GNN Embedding, quy trình tương tự, chạy notebook
[`GNN Embedding`](./notebooks/otto-recsys-gnn-embedding-generation.ipynb)

### Bước 2: Huấn luyện Mô hình Re-ranker và dự đoán
*   **Mục đích:** Sử dụng các tài sản đã tạo ở Bước 1 để xây dựng pipeline tạo ra đầu ra trực tiếp để submit lên kaggle
*   **Notebook:** [`Full Pipeline`](./notebooks/otto-recsys-full-pipeline.ipynb)

*   **Đầu ra:** File `submission.csv`