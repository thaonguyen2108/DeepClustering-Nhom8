# Hệ thống phân cụm khách hàng tự động

Đây là project đồ án dùng cho hai môn **Deep Learning** và **Khai phá dữ liệu** với đề tài:

**Hệ thống phân cụm khách hàng tự động sử dụng Deep Clustering dựa trên Autoencoder**

## Mục tiêu

- Cho phép người dùng đưa tập dữ liệu khách hàng vào hệ thống.
- Kiểm tra chất lượng dữ liệu đầu vào: missing value, duplicate, outlier, cột hằng, cột có độ duy nhất cao.
- Trực quan hóa dữ liệu phục vụ EDA/Khai phá dữ liệu.
- Đề xuất và cho phép người dùng xác nhận phương án tiền xử lý.
- Huấn luyện Autoencoder để học vector tiềm ẩn.
- Dùng vector tiềm ẩn để chọn số cụm K và phân cụm bằng KMeans.
- Đánh giá kết quả bằng Silhouette Score, Davies-Bouldin Index và Calinski-Harabasz Score.
- Xuất kết quả phân cụm và báo cáo xử lý dữ liệu.

## Chức năng chính

- Streamlit app hỗ trợ upload CSV/XLSX/JSON hoặc nhập đường dẫn dataset.
- Data profiling: kiểu cột, missing, duplicate, unique count, numeric/categorical/text.
- Biểu đồ EDA: Histogram, Boxplot, Correlation Heatmap, Scatterplot, Count Plot, Missing Value Chart.
- Phân loại cột đặc trưng và metadata/identifier, ví dụ Customer ID không dùng để train Autoencoder/KMeans.
- Tiền xử lý có xác nhận: missing, duplicate, outlier IQR/Z-score, encoding, scaling, text embedding.
- Pipeline Deep Clustering: preprocessing → embedding/vector đặc trưng → Autoencoder → vector tiềm ẩn → chọn K → KMeans → evaluation → visualization → export.
- Export dòng dữ liệu bị loại, báo cáo tiền xử lý, kết quả phân cụm và final report.

## Cấu trúc pipeline

```text
Dữ liệu đầu vào
  -> Kiểm tra dữ liệu
  -> Tiền xử lý
  -> Sinh embedding / vector đặc trưng
  -> Train Autoencoder
  -> Trích xuất vector tiềm ẩn
  -> Tự động chọn số cụm K
  -> KMeans
  -> Đánh giá
  -> Trực quan hóa
  -> Xuất kết quả
```

## Cài đặt

Nên dùng virtual environment của project nếu đã có:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu cần tạo mới:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy ứng dụng

```powershell
streamlit run app.py
```

Hoặc:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Quy trình sử dụng 7 tab

1. `Tải dữ liệu`: upload file hoặc nhập đường dẫn dataset.
2. `Tổng quan & trực quan hóa dữ liệu`: xem thống kê và biểu đồ EDA.
3. `Kiểm tra & tiền xử lý dữ liệu`: xem gợi ý xử lý, phân loại feature/metadata, xác nhận trước khi chạy pipeline.
4. `Cấu hình mô hình`: chỉnh Autoencoder, latent_dim, epochs, batch_size, learning_rate và số cụm K.
5. `Tiến trình xử lý`: chạy pipeline Deep Clustering và theo dõi trạng thái từng stage.
6. `Đánh giá kết quả`: xem metric, loss Autoencoder, biểu đồ chọn K và latent space.
7. `Kết quả phân cụm`: xem bảng khách hàng kèm nhãn cụm, cluster profiling và tải CSV.

## Output

Hệ thống tự tạo các thư mục khi cần:

```text
outputs/
  removed_rows/
  reports/
  clustering_results/
```

File output thường gặp:

- `outputs/removed_rows/removed_rows_<timestamp>.csv`
- `outputs/reports/data_cleaning_report_<timestamp>.json`
- `outputs/clustering_results/customer_segments_<timestamp>.csv`
- `outputs/reports/final_report_<timestamp>.json`

## Test nhanh

Kiểm tra compile:

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py pipeline\data_quality.py pipeline\__init__.py
```

Kiểm tra import:

```powershell
.\.venv\Scripts\python.exe -c "import app; import pipeline; print('import ok')"
```

## Lưu ý

- Dataset lớn có thể làm embedding, Autoencoder hoặc t-SNE chạy lâu. App có sample khi vẽ biểu đồ EDA.
- Text embedding dùng `sentence-transformers`; nếu model chưa có cache, lần đầu có thể cần tải model.
- Các cột metadata/identifier vẫn được giữ trong bảng kết quả để đối chiếu, nhưng không được dùng để train Autoencoder/KMeans.
- Nên giữ `random_state`/`random_seed` để kết quả dễ tái lập.
