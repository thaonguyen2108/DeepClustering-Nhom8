# CẤU HÌNH HỆ THỐNG DEEP CLUSTERING

## 1. Thông tin chung

Tên đề tài: **Hệ thống phân cụm khách hàng tự động sử dụng Deep Clustering dựa trên Autoencoder**.

Mục tiêu của hệ thống là cho phép người dùng đưa dữ liệu khách hàng vào ứng dụng, kiểm tra chất lượng dữ liệu, tiền xử lý dữ liệu, học biểu diễn tiềm ẩn bằng Autoencoder, phân cụm bằng KMeans, đánh giá kết quả và xuất báo cáo/kết quả phân cụm.

Hệ thống phục vụ hai hướng:

- **Khai phá dữ liệu**: kiểm tra dữ liệu, profiling, trực quan hóa EDA, phát hiện missing value, duplicate, outlier, high-cardinality và metadata/identifier.
- **Deep Learning**: huấn luyện Autoencoder để nén vector đặc trưng thành vector tiềm ẩn, sau đó dùng latent vector cho clustering.

Pipeline tổng thể:

```text
Dữ liệu đầu vào
→ Kiểm tra chất lượng dữ liệu
→ Tiền xử lý dữ liệu
→ Mã hóa categorical / chuẩn hóa numeric / embedding nếu có text
→ Autoencoder
→ Latent vector
→ KMeans
→ Đánh giá
→ Trực quan hóa
→ Xuất kết quả phân cụm
```

## 2. Cấu trúc thư mục và vai trò file

Các file/thư mục chính trong project hiện tại:

- `app.py`
  - Vai trò: giao diện Streamlit chính, điều phối 7 tab, lưu `st.session_state`, gọi các module trong `pipeline/`.
  - Các hàm UI chính: `render_load_tab()`, `render_eda_tab()`, `render_preprocessing_tab()`, `render_model_config_tab()`, `render_progress_tab()`, `render_evaluation_tab()`, `render_results_tab()`.
  - Hàm chạy pipeline chính: `run_pipeline()`.

- `pipeline/data_quality.py`
  - Vai trò: profiling dữ liệu, phân loại feature/metadata, phát hiện outlier IQR, đề xuất tiền xử lý, áp dụng cleaning plan, export removed rows, export cleaning report, export kết quả phân cụm và final report.
  - Các hàm chính: `profile_data()`, `recommend_preprocessing()`, `estimate_cleaning_impact()`, `apply_preprocessing_plan()`, `classify_column_role()`, `apply_column_role_overrides()`.

- `pipeline/ingestion.py`
  - Vai trò: đọc dữ liệu dạng bảng từ CSV, Excel hoặc JSON.
  - Class cấu hình: `IngestionConfig`.
  - Hàm public: `load_tabular_dataset()`.

- `pipeline/schema_detection.py`
  - Vai trò: suy luận kiểu cột `numeric`, `categorical`, `text`, `unknown`.
  - Class cấu hình: `SchemaDetectionConfig`.
  - Hàm public: `detect_schema()`.

- `pipeline/preprocessing.py`
  - Vai trò: chuẩn hóa numeric, mã hóa categorical, chuẩn bị text payload, tạo feature matrix.
  - Class cấu hình: `PreprocessingConfig`.
  - Hàm public: `preprocess_features()`.

- `pipeline/embedding.py`
  - Vai trò: sinh embedding cho cột text tự do nếu có.
  - Class cấu hình: `EmbeddingConfig`.
  - Hàm public: `embed_text_features()`.

- `Embeddings/embeddings.py`
  - Vai trò: engine thật dùng `sentence-transformers`, có xử lý batch size và GPU/CPU.
  - Class chính: `EmbeddingEngine`.

- `pipeline/autoencoder.py`
  - Vai trò: cấu hình, train Autoencoder bằng PyTorch, trích xuất `Z_latent`.
  - Class cấu hình: `AutoencoderConfig`.
  - Hàm public: `train_autoencoder_features()`.

- `pipeline/clustering.py`
  - Vai trò: chạy KMeans trên latent vector, thử nhiều K, chọn K tốt.
  - Class cấu hình: `ClusteringConfig`.
  - Hàm public: `cluster_latent_features()`.

- `pipeline/evaluation.py`
  - Vai trò: tính metric phân cụm và chuẩn bị dữ liệu visualization t-SNE.
  - Class cấu hình: `EvaluationConfig`.
  - Hàm public: `evaluate_clustering()`.

- `pipeline/__init__.py`
  - Vai trò: export các config, result, module và helper để `app.py` import tập trung.

- `README.md`
  - Vai trò: hướng dẫn cài đặt, chạy app, quy trình dùng 7 tab và output.

- `DataSet/`
  - Vai trò: chứa dataset mẫu/thật để test, gồm `shopping_behavior_updated.csv`, `customer_shopping_data.csv`, `online_retail_II.csv`.

- `outputs/`
  - Vai trò: chứa kết quả sinh ra khi chạy hệ thống, ví dụ `reports/`, `clustering_results/`, `removed_rows/`.

- `pipeline/visualization.py`
  - File này **không tồn tại** trong project hiện tại. Logic visualization đang nằm trong `app.py`, chủ yếu ở `render_eda_tab()`, `render_evaluation_tab()` và `render_latent_scatter()`.

## 3. Cấu hình giao diện Streamlit

Giao diện có 7 tab, được tạo trong hàm `main()` của `app.py`.

### 3.1. Tab Tải dữ liệu

- Hàm liên quan: `render_load_tab()` trong `app.py`.
- Mục đích: tải dữ liệu khách hàng vào hệ thống bằng upload file hoặc nhập đường dẫn.
- Người dùng thao tác: chọn file CSV/XLSX/JSON hoặc nhập đường dẫn file.
- Kết quả đầu ra: lưu dữ liệu vào `st.session_state`, hiển thị số dòng, số cột, kiểu cột, preview dữ liệu.
- Hàm đọc nguồn dữ liệu: `load_source()` trong `app.py`, gọi `load_tabular_dataset()` từ `pipeline/ingestion.py`.

### 3.2. Tab Tổng quan & trực quan hóa dữ liệu

- Hàm liên quan: `render_eda_tab()` trong `app.py`.
- Mục đích: khám phá dữ liệu ban đầu bằng thống kê và biểu đồ.
- Người dùng thao tác: chọn cột numeric/categorical để xem histogram, boxplot, heatmap, scatterplot, count plot, missing chart.
- Kết quả đầu ra: biểu đồ EDA và bảng profile theo cột.

### 3.3. Tab Kiểm tra & tiền xử lý dữ liệu

- Hàm liên quan: `render_preprocessing_tab()` trong `app.py`.
- Mục đích: kiểm tra missing, duplicate, outlier, phân loại feature/metadata, cho phép người dùng chỉnh kế hoạch xử lý.
- Người dùng thao tác: chọn vai trò cột `feature`/`metadata`, chọn cách xử lý missing/outlier/encoding/scaling/text, bấm xác nhận.
- Kết quả đầu ra: `current_df` đã được tiền xử lý, report tiền xử lý, removed rows nếu có.
- Logic xử lý chính nằm trong `pipeline/data_quality.py`.

### 3.4. Tab Cấu hình mô hình

- Hàm liên quan: `render_model_config_tab()` trong `app.py`.
- Mục đích: cấu hình Autoencoder và KMeans.
- Người dùng thao tác: chọn latent dimension tự động/thủ công, epochs, batch size, learning rate, random seed, chọn K tự động hoặc thủ công.
- Kết quả đầu ra: lưu cấu hình vào `st.session_state["model_config"]`.
- Helper chống lỗi K quá lớn: `prepare_k_widget_state()` trong `app.py`.

### 3.5. Tab Tiến trình xử lý

- Hàm liên quan: `render_progress_tab()` và `run_pipeline()` trong `app.py`.
- Mục đích: chạy pipeline và hiển thị tiến trình.
- Người dùng thao tác: bấm nút `Chạy pipeline Deep Clustering`.
- Kết quả đầu ra: bảng trạng thái stage, log, kết quả pipeline trong `st.session_state["pipeline_results"]`.

### 3.6. Tab Đánh giá kết quả

- Hàm liên quan: `render_evaluation_tab()` trong `app.py`.
- Mục đích: hiển thị metric clustering, biểu đồ chọn K, loss Autoencoder, latent space.
- Kết quả đầu ra: metric cards, K selection chart, cluster size chart, Autoencoder loss curve, t-SNE latent visualization.

### 3.7. Tab Kết quả phân cụm

- Hàm liên quan: `render_results_tab()` trong `app.py`.
- Mục đích: hiển thị dữ liệu kèm `cluster_label`, thống kê cụm, cluster profiling và tải CSV.
- Kết quả đầu ra: bảng kết quả, thống kê số lượng theo cụm, insight đơn giản và file CSV tải về.

## 4. Cấu hình đọc dữ liệu

Hệ thống hỗ trợ các định dạng trong `FILE_FORMAT_MAP` của `app.py`:

- `.csv` → `csv`
- `.xlsx` → `xlsx`
- `.json` → `json`

Trong `pipeline/ingestion.py`, `IngestionConfig` hỗ trợ:

- CSV: nhiều encoding candidate như `utf-8`, `utf-8-sig`, `cp1252`, `latin-1`; nhiều delimiter candidate như dấu phẩy, chấm phẩy, tab, `|`.
- Excel: engine candidate mặc định `openpyxl`.
- JSON: có `json_lines`, `json_orient`, `json_encoding`.

Dữ liệu sau khi đọc được lưu vào `st.session_state` trong `reset_for_new_dataset()` của `app.py`:

- `raw_df`: dữ liệu gốc có thêm `original_index`.
- `current_df`: dữ liệu hiện tại sau các bước xử lý.
- `original_file_name`: tên file/nguồn dữ liệu.
- `data_profile`: kết quả profiling từ `profile_data()`.
- `feature_columns`, `metadata_columns`, `column_role_reasons`: kết quả phân loại cột.

Xử lý lỗi đọc file:

- `load_source()` trong `app.py` dùng `try/except`.
- Nếu file không tồn tại, định dạng không hỗ trợ, hoặc đọc lỗi, hệ thống trả thông báo lỗi thân thiện thay vì traceback thô.

## 5. Cấu hình kiểm tra chất lượng dữ liệu

Logic chính nằm trong `pipeline/data_quality.py`.

### 5.1. Missing values

- Mục đích: phát hiện ô dữ liệu bị thiếu.
- Rule: `profile_data()` tính `total_missing_cells`, `missing_count`, `missing_ratio` theo từng cột.
- Gợi ý xử lý trong `recommend_preprocessing()`:
  - Numeric: mặc định `median` nếu có outlier, `mean` nếu không có outlier; nếu missing rất ít có thể gợi ý `drop_rows`.
  - Categorical: `mode` nếu missing ít, `unknown` nếu missing nhiều hơn.
  - Text: mặc định `empty`.
- Khi áp dụng plan, `apply_preprocessing_plan()` thực hiện impute hoặc đánh dấu drop row.

### 5.2. Duplicate rows

- Mục đích: phát hiện bản ghi trùng.
- Rule: dùng `DataFrame.duplicated()` sau khi bỏ `original_index`.
- Gợi ý mặc định: `duplicates = "drop"` trong `recommend_preprocessing()`.
- Nếu drop, `apply_preprocessing_plan()` đưa dòng bị loại vào removed rows report.

### 5.3. Outlier

- Mục đích: phát hiện giá trị numeric bất thường.
- Rule chính: IQR trong `detect_outlier_masks()`.
- IQR tính theo Q1, Q3, `IQR = Q3 - Q1`; outlier nếu nhỏ hơn `Q1 - 1.5*IQR` hoặc lớn hơn `Q3 + 1.5*IQR`.
- Plan hỗ trợ:
  - `keep`
  - `cap_iqr`
  - `drop_iqr`
  - `drop_zscore`
- Khi cap/drop, thao tác được ghi vào cleaning report.

### 5.4. Constant / near-constant columns

- Mục đích: phát hiện cột ít giá trị, không có nhiều thông tin phân cụm.
- Rule trong `profile_data()`:
  - `is_constant`: `unique_count <= 1`.
  - `is_near_constant`: top value chiếm từ 98% số dòng trở lên.
- Plan có option `drop_constant_columns`.

### 5.5. High-cardinality columns

- Mục đích: cảnh báo cột categorical có quá nhiều giá trị khác nhau.
- Trong `PreprocessingConfig`, `max_onehot_categories = 100`; nếu vượt ngưỡng, preprocessing fallback sang label/ordinal encoder.
- Trong `pipeline/data_quality.py`, cột chuỗi có unique ratio rất cao có thể được phân loại là metadata nghi vấn.

## 6. Cấu hình lựa chọn feature và metadata

Feature columns là các cột được dùng để tạo feature matrix, train Autoencoder và chạy KMeans. Metadata/identifier columns là các cột chỉ dùng để đối chiếu, không dùng để học mô hình.

Logic nằm trong:

- `classify_column_role()` ở `pipeline/data_quality.py`.
- `apply_column_role_overrides()` ở `pipeline/data_quality.py`.
- UI override nằm trong `render_preprocessing_tab()` của `app.py`.

Rule hiện tại:

- `original_index` luôn là metadata.
- Tên cột giống mã định danh hoặc kết thúc bằng `_id` được xem là metadata, ví dụ `customer_id`, `user_id`, `order_id`.
- Các tên đặc biệt trong code gồm: `id`, `customerid`, `customer_id`, `user_id`, `client_id`, `invoice`, `order_id`.
- Cột ngày/giờ dạng thô có chứa `date`, `time`, `datetime`, `timestamp` được xem là metadata nếu chưa trích xuất đặc trưng thời gian.
- Cột categorical/text có `unique_ratio >= 0.90` và nhiều giá trị duy nhất được xem là suspicious high-uniqueness, có khả năng là mã định danh hoặc metadata.
- Numeric unique cao như `Age`, `Income`, `Purchase Amount` không tự động bị loại chỉ vì unique ratio cao; nếu là numeric và không có tên định danh thì vẫn có thể là feature.

`unique_ratio` được tính bằng:

```text
unique_count / số giá trị non-null
```

Người dùng có thể override vai trò từng cột trong tab `Kiểm tra & tiền xử lý dữ liệu`. Nếu chọn `metadata`, cột không đi vào Autoencoder/KMeans. Nếu chọn `feature`, cột sẽ đi vào feature matrix nếu còn tồn tại sau cleaning.

Metadata vẫn được giữ trong bảng kết quả và file `customer_segments_<timestamp>.csv` để đối chiếu khách hàng/bản ghi.

## 7. Cấu hình tiền xử lý dữ liệu

Logic chính nằm trong `pipeline/preprocessing.py`.

### 7.1. Numeric columns

- Numeric được chuẩn hóa bằng scaler.
- `PreprocessingConfig.numeric_scaler` mặc định là `standard`.
- Scaler hỗ trợ trong code:
  - `StandardScaler`
  - `MinMaxScaler`
  - `RobustScaler`
- Trong UI, người dùng chọn scaling ở `render_preprocessing_tab()` của `app.py`; `_map_scaler()` map option UI sang config thực tế.

### 7.2. Categorical columns

- Categorical dạng chữ không đưa trực tiếp vào Autoencoder.
- `PreprocessingConfig.categorical_encoder` mặc định là `onehot`.
- Encoder hỗ trợ:
  - `OneHotEncoder`
  - `OrdinalEncoder`/label encoding
- Nếu cardinality cao hơn `max_onehot_categories = 100`, hệ thống fallback sang `high_cardinality_fallback_encoder = "label"`.

### 7.3. Text columns

- Text tự do được tách thành `X_text`.
- Nếu `cleaning_plan["text"] == "use"`, `PreprocessingConfig.include_text_columns=True`, text có thể đi qua embedding.
- Nếu không có text column hoặc người dùng chọn ignore/drop, embedding trả ma trận rỗng hợp lệ.

### 7.4. Feature matrix cuối cùng

Autoencoder chỉ nhận dữ liệu dạng số. Feature matrix được ghép từ:

- Numeric scaled features.
- Categorical encoded features.
- Text embeddings nếu có text tự do và bật embedding.

Trong `run_pipeline()` của `app.py`, hệ thống tạo `feature_df = current_df[feature_columns]`, nghĩa là metadata không đi vào preprocessing, Autoencoder hoặc KMeans.

## 8. Cấu hình Autoencoder

Autoencoder nằm trong `pipeline/autoencoder.py`.

Sơ đồ train:

```text
X_preprocessed
→ Encoder
→ Z_latent
→ Decoder
→ X_reconstructed
```

Sau khi train:

```text
X_preprocessed
→ Encoder
→ Z_latent
→ KMeans
```

Input của Autoencoder là feature matrix dạng số sau preprocessing và embedding. Target khi train là chính input đó, vì Autoencoder học tái tạo dữ liệu đầu vào. Loss mặc định trong `AutoencoderConfig` là `mse`, tức Mean Squared Error / reconstruction loss.

Class cấu hình: `AutoencoderConfig` trong `pipeline/autoencoder.py`.

Các cấu hình chính:

- `epochs`: mặc định 50.
- `batch_size`: mặc định 128.
- `learning_rate`: mặc định 0.001.
- `latent_dim`: mặc định `"auto"`, nhưng `app.py` có UI cho chế độ tự động đề xuất hoặc tự nhập.
- `min_latent`: 8.
- `max_latent`: 128.
- `hidden_dims`: nếu `None`, hệ thống tự chọn kiến trúc.
- `max_hidden_layers`: 3.
- `activation`: `relu`.
- `loss`: `mse`.
- `optimizer`: `adam`.
- `use_gpu`: `True`.
- `random_seed`: 42.
- `handle_non_finite`: `replace`.

Code train chính:

- `train_autoencoder_features()` trong `pipeline/autoencoder.py`.
- `AutoencoderModule.run()` điều phối chuẩn bị dữ liệu, chọn kiến trúc, train, encode latent.
- `_train_model()` train PyTorch model.
- `_encode_latent()` lấy vector tiềm ẩn sau Encoder.
- `_build_training_metadata()` lưu `loss_history`, `final_loss`, `latent_shape`.

Trong `app.py`, cấu hình Autoencoder nằm ở `render_model_config_tab()` và được dùng trong `run_pipeline()` khi tạo `AutoencoderConfig(...)`.

Latent dimension trong UI:

- Tự động đề xuất trong `render_model_config_tab()` theo logic:
  - `default_latent_dim = min(8, max(2, input_dim_estimate // 4))`.
  - Nếu `input_dim_estimate > 2`, đảm bảo latent nhỏ hơn input khi có thể.
- Cho phép người dùng nhập thủ công.
- Có cảnh báo nếu `latent_dim >= input_dim`.

Kiến trúc hidden layer tự động nằm trong `AutoencoderModule._select_architecture()` và `_resolve_hidden_dims()` ở `pipeline/autoencoder.py`. Tài liệu này không trích toàn bộ công thức vì logic có nhiều nhánh kiểm tra/fallback; khi báo cáo có thể nói hệ thống tự chọn hidden dimensions dựa trên `input_dim`, `latent_dim`, `max_hidden_layers` và có fallback architecture nếu kiến trúc chính không phù hợp.

## 9. Cấu hình embedding / feature vector

Embedding nằm trong:

- `pipeline/embedding.py`
- `Embeddings/embeddings.py`

Class cấu hình: `EmbeddingConfig` trong `pipeline/embedding.py`.

Các cấu hình chính:

- `batch_size`: 32.
- `merge_strategy`: `concat_text`.
- `handle_empty`: `replace`.
- `default_text`: `unknown`.
- `model_name`: `intfloat/multilingual-e5-small`.
- `normalize`: `True`.
- `auto_batch`: `True`.
- `force_offline`: `True`.

Embedding được dùng khi có text column tự do và người dùng chọn dùng text embedding. Nếu không có text column, `embed_text_features()` vẫn trả về `X_embedding` rỗng hợp lệ để pipeline không bị lỗi.

Cần phân biệt:

- Categorical ngắn như `Gender`, `Category`, `Payment Method`, `Size` đi qua encoding, không đi qua sentence embedding.
- Text tự do dài hoặc mô tả dạng câu mới phù hợp với embedding.

Feature vector cuối cùng cho Autoencoder gồm `X_numeric` từ preprocessing và `X_embedding` nếu có.

## 10. Cấu hình clustering

Clustering nằm trong `pipeline/clustering.py`.

Thuật toán chính: **KMeans**.

Dữ liệu đầu vào của KMeans là `Z_latent`, tức vector tiềm ẩn sau Encoder, không phải raw data.

Class cấu hình: `ClusteringConfig`.

Các cấu hình chính:

- `k_min`: mặc định 2.
- `k_max`: mặc định 10.
- `random_state`: 42.
- `n_init`: 10.
- `max_iter`: 300.
- `algorithm`: `lloyd`.
- `silhouette_metric`: `euclidean`.
- `prefer_elbow_when_close`: `True`.
- `elbow_validation_tolerance`: 0.02.

Code liên quan:

- `cluster_latent_features()` là hàm public.
- `ClusteringModule._evaluate_k_candidates()` thử các giá trị K.
- `ClusteringModule._fit_kmeans_and_score()` fit KMeans và tính silhouette/inertia.
- `ClusteringModule._select_best_k()` chọn K tốt, có xét Silhouette và elbow khi gần nhau.

Trong UI, K được cấu hình ở `render_model_config_tab()` của `app.py`.

Xử lý dữ liệu quá ít dòng:

- `prepare_k_widget_state(n_samples)` trong `app.py` kiểm tra nếu `n_samples < 3` thì không render input K và cảnh báo người dùng.
- Hàm này cũng clamp `k_min`, `k_max`, `manual_k` để tránh lỗi `value > max_value` của Streamlit.

Baseline KMeans trên feature preprocessing hiện chưa có trong UI chính. Tuy nhiên đã có test/report gần nhất so sánh KMeans trên feature preprocessing với KMeans trên latent vector trong các file `outputs/reports/dataset_evaluation_summary*.json`.

## 11. Cấu hình evaluation

Evaluation nằm trong `pipeline/evaluation.py`.

Class cấu hình: `EvaluationConfig`.

Các cấu hình chính:

- `compute_tsne`: `True`.
- `perplexity`: 30.0.
- `random_state`: 42.
- `tsne_init`: `random`.
- `tsne_metric`: `euclidean`.
- `compute_davies_bouldin`: `True`.
- `compute_calinski_harabasz`: `True`.

Các chỉ số:

- **Silhouette Score**
  - Ý nghĩa: đo mức độ điểm dữ liệu gần cụm của nó và xa cụm khác.
  - Càng cao càng tốt.
  - Dùng trong clustering để chọn K và trong evaluation để báo cáo.

- **Davies-Bouldin Index**
  - Ý nghĩa: đo mức độ cụm tách biệt và compact.
  - Càng thấp càng tốt.
  - Tính sau khi đã có nhãn cụm.

- **Calinski-Harabasz Score**
  - Ý nghĩa: đo tỷ lệ phân tán giữa cụm so với trong cụm.
  - Càng cao càng tốt.
  - Tính sau khi đã có nhãn cụm.

- **Inertia**
  - Ý nghĩa: tổng khoảng cách bình phương từ điểm tới tâm cụm.
  - Càng thấp khi K tăng; dùng để vẽ Elbow/Inertia trong clustering metadata.

Code liên quan:

- `evaluate_clustering()` trong `pipeline/evaluation.py`.
- `_compute_metrics()` tính Silhouette, DBI, CH.
- `_build_visualization_data()` chuẩn bị t-SNE nếu bật.

Metric chính để chọn K trong `pipeline/clustering.py` là Silhouette Score, có tham khảo elbow khi điểm gần nhau.

## 12. Cấu hình visualization

Visualization hiện nằm trong `app.py`, chưa tách thành `pipeline/visualization.py`.

Các biểu đồ hiện có:

- Histogram
  - Tab: `Tổng quan & trực quan hóa dữ liệu`.
  - Hàm: `render_eda_tab()`.
  - Mục đích: xem phân phối một cột numeric.

- Boxplot
  - Tab: `Tổng quan & trực quan hóa dữ liệu`.
  - Hàm: `render_eda_tab()`.
  - Mục đích: xem median, Q1-Q3, outlier.

- Correlation heatmap
  - Tab: `Tổng quan & trực quan hóa dữ liệu`.
  - Hàm: `render_eda_tab()`.
  - Mục đích: xem tương quan giữa các cột numeric feature.

- Scatterplot
  - Tab: `Tổng quan & trực quan hóa dữ liệu`.
  - Hàm: `render_eda_tab()`.
  - Mục đích: xem quan hệ giữa hai biến số.

- Count plot / bar chart
  - Tab: `Tổng quan & trực quan hóa dữ liệu`.
  - Hàm: `render_eda_tab()`.
  - Mục đích: xem tần suất từng nhóm/category.

- Missing value chart
  - Tab: `Tổng quan & trực quan hóa dữ liệu`.
  - Hàm: `render_eda_tab()`.
  - Mục đích: xem số missing theo cột.

- Silhouette theo K và Elbow/Inertia
  - Tab: `Đánh giá kết quả`.
  - Hàm: `render_evaluation_tab()`.
  - Dữ liệu: `clustering_result.clustering_metadata`.

- Cluster size distribution
  - Tab: `Đánh giá kết quả` và `Kết quả phân cụm`.
  - Hàm: `render_evaluation_tab()`, `render_results_tab()`.

- Autoencoder loss curve
  - Tab: `Đánh giá kết quả`.
  - Hàm: `render_evaluation_tab()`.
  - Dữ liệu: `autoencoder_result.training_metadata["loss_history"]`.

- Latent space visualization
  - Tab: `Đánh giá kết quả`.
  - Hàm: `render_latent_scatter()` và `render_evaluation_tab()`.
  - Dữ liệu: `evaluation_result.visualization_data["Z_2d"]`.

Biểu đồ phân rã/decomposition không xuất hiện trong UI chính vì project hiện không có module xử lý chuỗi thời gian phù hợp; đây là lựa chọn đúng với customer segmentation hiện tại.

Text giải thích biểu đồ nằm trong dictionary `CHART_HELP` ở `app.py`.

## 13. Cấu hình export/report

Logic export nằm trong `pipeline/data_quality.py`.

Các thư mục output:

- `outputs/removed_rows/`
- `outputs/reports/`
- `outputs/clustering_results/`

Các file:

- `outputs/removed_rows/removed_rows_<timestamp>.csv`
  - Tạo khi có dòng bị drop.
  - Nội dung gồm `original_index`, `reason`, `action`, `affected_column` và dữ liệu gốc của dòng bị loại.
  - Hàm export: `export_removed_rows()`.

- `outputs/reports/data_cleaning_report_<timestamp>.json`
  - Tạo khi áp dụng preprocessing plan.
  - Nội dung gồm thời gian tạo, số dòng trước/sau, số dòng bị loại, cột bị drop, actions, plan.
  - Hàm export: `export_cleaning_report()`.

- `outputs/clustering_results/customer_segments_<timestamp>.csv`
  - Tạo sau khi pipeline phân cụm xong.
  - Nội dung gồm dữ liệu sau xử lý kèm `cluster_label`; metadata columns vẫn được giữ để đối chiếu.
  - Hàm export: `export_cluster_results()`.

- `outputs/reports/final_report_<timestamp>.json`
  - Tạo sau pipeline.
  - Nội dung gồm dataset summary, `feature_columns`, `metadata_columns`, suspicious high-uniqueness columns, preprocessing actions, Autoencoder config, input_dim, latent_dim, compression ratio, selected K, metrics, output paths.
  - Hàm export: `export_final_report()`.

Trong `app.py`, final report được tạo trong `run_pipeline()`.

## 14. Cấu hình session_state

Các key chính được tạo trong `init_session_state()` của `app.py`:

- `raw_df`
  - Dữ liệu gốc sau khi đọc, có `original_index`.
  - Tạo trong `reset_for_new_dataset()`.
  - Dùng ở tab tải dữ liệu, EDA, tiền xử lý.

- `current_df`
  - Dữ liệu hiện tại sau tiền xử lý.
  - Ban đầu bằng `raw_df`, sau xác nhận preprocessing sẽ là `cleaning_result.cleaned_dataframe`.
  - Dùng trong cấu hình mô hình, tiến trình xử lý, pipeline.

- `original_file_name`
  - Tên file/nguồn dữ liệu.
  - Tạo khi đọc dataset.

- `data_profile`
  - Kết quả từ `profile_data()`.
  - Dùng cho EDA, tiền xử lý, feature/metadata selection.

- `cleaning_plan`
  - Kế hoạch xử lý missing/duplicate/outlier/encoding/scaling/text.
  - Tạo từ `recommend_preprocessing()`, được cập nhật theo lựa chọn UI.

- `cleaning_impact`
  - Ước lượng số dòng/cột bị ảnh hưởng trước khi xác nhận.
  - Tạo từ `estimate_cleaning_impact()`.

- `cleaning_result`
  - Kết quả sau khi áp dụng preprocessing plan.
  - Chứa `cleaned_dataframe`, `removed_rows`, `report`, paths.

- `cleaning_confirmed`
  - Boolean cho biết người dùng đã xác nhận tiền xử lý hay chưa.
  - Nếu chưa xác nhận, tab tiến trình không cho chạy pipeline chính.

- `pipeline_results`
  - Dict chứa kết quả từng stage: schema, preprocessing, embedding, autoencoder, clustering, evaluation, result_frame.

- `pipeline_logs`
  - Danh sách log text từ các stage pipeline.

- `pipeline_status`
  - Trạng thái từng stage: tên stage, status, start/end time, duration, note.

- `pipeline_error`
  - Lỗi pipeline nếu có.

- `output_paths`
  - Đường dẫn file output đã sinh ra.

- `model_config`
  - Cấu hình Autoencoder/KMeans/visualization do người dùng chọn.

- `feature_columns`
  - Cột dùng làm feature matrix.

- `metadata_columns`
  - Cột chỉ dùng để đối chiếu/export.

- `column_role_reasons`
  - Lý do phân loại từng cột.

- `column_role_overrides`
  - Lựa chọn feature/metadata do người dùng override.

Các key ví dụ như `preprocessed_df`, `feature_matrix`, `latent_vectors`, `cluster_labels`, `metrics`, `selected_k` không tồn tại riêng trong `st.session_state`; thông tin tương ứng nằm trong `pipeline_results`.

## 15. Cấu hình kiểm soát lỗi và cảnh báo

- Lỗi đọc file:
  - Xử lý trong `load_source()` của `app.py` bằng `try/except`.
  - Không hiển thị traceback thô cho người dùng.

- Dữ liệu quá ít dòng để clustering:
  - Xử lý trong `prepare_k_widget_state()` của `app.py`.
  - Nếu `n_samples < 3`, không render input K và cảnh báo cần ít nhất 3 dòng.

- Không có text column:
  - `embed_text_features()` vẫn trả `X_embedding` rỗng hợp lệ.
  - UI không bắt buộc dùng text embedding.

- Không có numeric/categorical column:
  - `render_eda_tab()` hiển thị cảnh báo khi không đủ cột để vẽ từng loại biểu đồ.
  - `preprocess_features()` có logic xử lý block rỗng.

- `latent_dim >= input_dim`:
  - `render_model_config_tab()` cảnh báo rằng Autoencoder không thể hiện rõ vai trò nén đặc trưng.

- Warning `use_container_width`:
  - Đã thay bằng `width="stretch"` trong `app.py`.

- Warning Matplotlib boxplot labels/tick_labels:
  - Project hiện vẫn dùng `labels=` trong boxplot group. Nếu Streamlit/Matplotlib phiên bản mới cảnh báo, nên đổi sang `tick_labels=`. Chưa xác định đây là lỗi nghiêm trọng trong code hiện tại.

## 16. Luồng chạy tổng thể của hệ thống

1. Người dùng tải dữ liệu trong tab `Tải dữ liệu`.
2. Hệ thống đọc file bằng `load_source()` và `load_tabular_dataset()`.
3. Hệ thống thêm `original_index`, tạo `raw_df`, `current_df`.
4. Hệ thống profiling dữ liệu bằng `profile_data()`.
5. Hệ thống phân loại feature/metadata bằng `classify_column_role()`.
6. Hệ thống đề xuất tiền xử lý bằng `recommend_preprocessing()`.
7. Người dùng kiểm tra, override feature/metadata nếu cần, chỉnh plan và xác nhận.
8. Hệ thống áp dụng plan bằng `apply_preprocessing_plan()`.
9. Người dùng cấu hình Autoencoder/KMeans trong `render_model_config_tab()`.
10. Người dùng chạy pipeline trong tab `Tiến trình xử lý`.
11. `run_pipeline()` tạo `feature_df` từ `feature_columns`.
12. Hệ thống detect schema, preprocessing, embedding nếu có.
13. Hệ thống train Autoencoder và lấy `Z_latent`.
14. Hệ thống chọn K và chạy KMeans trên `Z_latent`.
15. Hệ thống tính metric evaluation.
16. Hệ thống chuẩn bị visualization.
17. Hệ thống export `customer_segments` và `final_report`.
18. Người dùng xem đánh giá và kết quả phân cụm.

## 17. Ví dụ cấu hình thực tế từ dataset đã test

Số liệu dưới đây lấy từ báo cáo test gần nhất trong `outputs/reports/dataset_evaluation_summary_after_date_metadata_fix.json`.

Dataset chính: `shopping_behavior_updated.csv`.

- Số dòng test: 3.900.
- Metadata column: `Customer ID`.
- Feature matrix shape: `(3900, 143)`.
- Autoencoder:
  - `input_dim`: 143.
  - `latent_dim`: 8.
  - Compression ratio: `143 → 8`.
  - Epochs test: 6.
  - Loss: `0.1088 → 0.0704`.
- KMeans trên feature preprocessing:
  - Best K: 2.
  - Silhouette: 0.0974.
- KMeans trên latent vector:
  - Selected K: 4.
  - Silhouette: 0.3742.
  - Davies-Bouldin Index: 0.9278.
  - Calinski-Harabasz Score: 2936.77.

Dataset so sánh: `customer_shopping_data.csv` sample 5.000 dòng.

- Metadata columns: `invoice_no`, `customer_id`, `invoice_date`.
- Feature matrix shape: `(5000, 26)`.
- Autoencoder:
  - `input_dim`: 26.
  - `latent_dim`: 6.
  - Compression ratio: `26 → 6`.
  - Loss: `0.2574 → 0.1612`.
- KMeans trên feature preprocessing:
  - Best K: 2.
  - Silhouette: 0.1956.
- KMeans trên latent vector:
  - Selected K: 3.
  - Silhouette: 0.4625.
  - Davies-Bouldin Index: 0.7468.
  - Calinski-Harabasz Score: 8099.31.

Nhận xét kỹ thuật: trong cả hai dataset, clustering trên latent vector tốt hơn rõ so với KMeans trực tiếp trên feature preprocessing theo Silhouette Score.

## 18. Ghi chú cho báo cáo/bảo vệ

- Nếu được hỏi Autoencoder nhận gì:
  - Autoencoder nhận feature matrix dạng số sau preprocessing. Numeric đã scale, categorical đã encode, text embedding được ghép nếu có.

- Nếu được hỏi vì sao không dùng KMeans trực tiếp:
  - Có thể dùng KMeans trực tiếp như baseline, nhưng Autoencoder học biểu diễn nén giúp giảm nhiễu và giảm chiều. Trong test gần nhất, latent vector cho Silhouette cao hơn baseline trên cả hai dataset.

- Nếu được hỏi categorical dạng chữ có đưa vào embedding không:
  - Không mặc định. Categorical ngắn như Gender, Category, Payment Method được mã hóa bằng OneHot/Ordinal. Embedding chỉ dùng cho text tự do có ý nghĩa ngữ nghĩa dài hơn.

- Nếu được hỏi Customer ID có dùng để train không:
  - Không. Customer ID là metadata/identifier, được giữ để đối chiếu kết quả nhưng không đưa vào Autoencoder hoặc KMeans.

- Nếu được hỏi hệ thống tự động ở đâu:
  - Tự động đọc và profile dữ liệu, tự phát hiện feature/metadata, tự đề xuất preprocessing, tự chọn K, tự train Autoencoder, tự tính metric và tự export report. Người dùng vẫn có quyền xác nhận và chỉnh cấu hình trước khi chạy pipeline chính.

## 19. Kiểm tra sau khi tạo file

File này chỉ là tài liệu, không thay đổi logic code. Các kiểm tra cần thực hiện sau khi tạo:

- Kiểm tra file `cauhinh.md` tồn tại ở thư mục gốc project.
- Kiểm tra nội dung không rỗng.
- Kiểm tra Markdown hiển thị đúng các heading, bullet và code block.
- Không cần chạy lại toàn bộ pipeline vì không sửa code logic.

