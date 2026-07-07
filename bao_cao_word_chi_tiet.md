# Nội dung báo cáo Word cho 2 môn

Đề tài: **Hệ thống phân cụm khách hàng tự động sử dụng Deep Clustering dựa trên Autoencoder**

Tài liệu này được viết dựa trên project hiện tại tại `A:\DoAn\DeepClustering\Project`, gồm ứng dụng Streamlit `app.py`, các module trong `pipeline/`, dữ liệu trong `DataSet/` và các report trong `outputs/reports/`. Những số liệu thực nghiệm chính được lấy từ `outputs/reports/dataset_evaluation_summary_after_date_metadata_fix.json`. Những nội dung chưa có đủ bằng chứng trong code được ghi rõ là cần kiểm tra lại trong project.

---

# I. Nội dung báo cáo Khai phá dữ liệu

## Chương 1. Tổng quan đề tài

### 1.1. Lý do chọn đề tài

Trong bối cảnh thương mại điện tử và bán lẻ hiện nay, dữ liệu khách hàng ngày càng lớn và đa dạng. Mỗi khách hàng có thể được mô tả bởi nhiều thông tin khác nhau như tuổi, giới tính, sản phẩm đã mua, danh mục sản phẩm, số tiền mua hàng, phương thức thanh toán, tần suất mua hàng hoặc địa điểm mua sắm. Nếu chỉ quan sát thủ công, doanh nghiệp khó nhận ra các nhóm khách hàng có hành vi giống nhau hoặc có xu hướng tiêu dùng khác nhau.

Phân cụm khách hàng là một bài toán quan trọng trong khai phá dữ liệu vì phần lớn dữ liệu khách hàng không có nhãn sẵn. Doanh nghiệp thường không biết trước khách hàng nào thuộc nhóm giá trị cao, nhóm mua thường xuyên hay nhóm nhạy cảm với khuyến mãi. Vì vậy, cần áp dụng các kỹ thuật khai phá dữ liệu không giám sát để phát hiện cấu trúc tiềm ẩn trong dữ liệu. Kết quả phân cụm có thể hỗ trợ marketing, chăm sóc khách hàng, xây dựng chương trình khuyến mãi và phân tích hành vi mua sắm.

Đề tài “Hệ thống phân cụm khách hàng tự động sử dụng Deep Clustering dựa trên Autoencoder” được chọn vì kết hợp được quy trình khai phá dữ liệu truyền thống với kỹ thuật học biểu diễn. Trong báo cáo Khai phá dữ liệu, trọng tâm không chỉ nằm ở Autoencoder mà nằm ở toàn bộ quy trình: tải dữ liệu, kiểm tra chất lượng dữ liệu, EDA, tiền xử lý, lựa chọn đặc trưng, phân cụm, đánh giá và diễn giải cụm.

### 1.2. Mục tiêu đề tài

Mục tiêu của đề tài theo hướng Khai phá dữ liệu là xây dựng một hệ thống hỗ trợ quy trình phân tích và phân cụm khách hàng từ dữ liệu dạng bảng. Hệ thống cho phép người dùng tải dữ liệu khách hàng, quan sát dữ liệu ban đầu, phát hiện vấn đề chất lượng dữ liệu, xác nhận phương án tiền xử lý, tạo ma trận đặc trưng, phân cụm khách hàng bằng KMeans và đánh giá chất lượng cụm bằng các chỉ số nội tại.

Cụ thể, hệ thống hướng đến các mục tiêu: hỗ trợ EDA và trực quan hóa dữ liệu; phát hiện missing value, duplicate, outlier, high-cardinality columns và metadata columns; cho phép người dùng kiểm tra và xác nhận preprocessing trước khi chạy pipeline; tự động chọn số cụm K; đánh giá cụm bằng Silhouette Score, Davies-Bouldin Index và Calinski-Harabasz Score; diễn giải cụm bằng cluster profiling trên các đặc trưng gốc; xuất kết quả phân cụm và report phục vụ lưu trữ.

### 1.3. Phạm vi đề tài

Đề tài tập trung vào bài toán phân cụm khách hàng trên dữ liệu dạng bảng. Hệ thống không giải quyết bài toán phân loại có nhãn, không dự đoán nhãn khách hàng có sẵn, không xây dựng hệ thống bán hàng hoàn chỉnh và chưa triển khai production. Phạm vi hiện tại phù hợp với dữ liệu customer segmentation, trong đó mỗi dòng là một khách hàng hoặc một giao dịch mua sắm đã được mô tả bằng các thuộc tính numeric và categorical.

### 1.4. Ý nghĩa thực tiễn

Hệ thống giúp người dùng chia khách hàng thành các nhóm có đặc điểm tương đồng. Từ đó, người phân tích có thể quan sát nhóm khách hàng có xu hướng chi tiêu cao hơn, nhóm mua nhiều lần hơn, nhóm có đánh giá tốt hơn hoặc nhóm phản ứng với các hình thức thanh toán, khuyến mãi khác nhau. Kết quả này không thay thế quyết định kinh doanh, nhưng cung cấp cơ sở dữ liệu ban đầu để doanh nghiệp thiết kế chiến lược marketing và chăm sóc khách hàng phù hợp hơn.

## Chương 2. Cơ sở lý thuyết

### 2.1. Khai phá dữ liệu

Khai phá dữ liệu là quá trình phát hiện tri thức, mẫu, xu hướng hoặc cấu trúc tiềm ẩn từ tập dữ liệu lớn. Trong đề tài này, khai phá dữ liệu được áp dụng để tìm ra các nhóm khách hàng tương đồng mà không cần nhãn sẵn. Dữ liệu đầu vào không chỉ được đưa thẳng vào mô hình mà phải trải qua các bước kiểm tra, làm sạch, biến đổi và đánh giá để kết quả phân cụm có ý nghĩa hơn.

### 2.2. Quy trình khai phá dữ liệu

Quy trình khai phá dữ liệu trong hệ thống gồm các bước: thu thập hoặc tải dữ liệu, profiling, EDA, kiểm tra chất lượng dữ liệu, tiền xử lý, lựa chọn feature/metadata, tạo feature matrix, học biểu diễn bằng Autoencoder, phân cụm bằng KMeans, đánh giá cụm, trực quan hóa và diễn giải kết quả. Trong app Streamlit, quy trình này được triển khai qua 7 tab: tải dữ liệu; tổng quan và trực quan hóa; kiểm tra và tiền xử lý; cấu hình mô hình; tiến trình xử lý; đánh giá kết quả; kết quả phân cụm.

### 2.3. Trực quan hóa dữ liệu

Hệ thống hỗ trợ các biểu đồ EDA trong `render_eda_tab()` của `app.py`. Histogram dùng để xem phân phối của biến số như Age, Purchase Amount hoặc Previous Purchases. Boxplot dùng để xem median, Q1, Q3 và outlier. Correlation heatmap dùng để xem tương quan giữa các biến numeric. Scatterplot giúp quan sát quan hệ giữa hai biến số. Count plot dùng cho dữ liệu categorical như Gender, Category, Payment Method. Missing value chart dùng để xem cột nào có nhiều giá trị thiếu.

Các biểu đồ này không tạo ra cụm, mà hỗ trợ người phân tích hiểu dữ liệu trước khi tiền xử lý và phân cụm. Ví dụ, nếu boxplot cho thấy một biến số có nhiều outlier, hệ thống có thể gợi ý cap IQR hoặc drop outlier. Nếu count plot cho thấy một cột categorical có quá nhiều giá trị khác nhau, hệ thống cảnh báo high cardinality để tránh mở rộng OneHot quá lớn.

### 2.4. Tiền xử lý dữ liệu

Tiền xử lý dữ liệu là bước bắt buộc trước khi train Autoencoder và KMeans. Hệ thống xử lý missing value bằng mean, median, mode, unknown hoặc drop rows tùy kiểu dữ liệu và kế hoạch người dùng xác nhận. Duplicate rows được phát hiện bằng `DataFrame.duplicated()` sau khi bỏ `original_index`. Outlier được phát hiện chủ yếu bằng IQR, hỗ trợ keep, cap IQR, drop IQR hoặc drop Z-score. Numeric columns được scale bằng StandardScaler, MinMaxScaler hoặc RobustScaler. Categorical columns được mã hóa bằng OneHotEncoder hoặc OrdinalEncoder; nếu cardinality vượt ngưỡng `max_onehot_categories = 100`, hệ thống fallback sang label/ordinal encoding.

### 2.5. Lựa chọn đặc trưng và metadata

Feature columns là các cột dùng để tạo feature matrix và đưa vào mô hình. Metadata columns là các cột chỉ dùng để đối chiếu kết quả, không dùng để train. Trong project, logic này nằm ở `classify_column_role()` và `apply_column_role_overrides()` trong `pipeline/data_quality.py`. Các cột như `Customer ID`, `customer_id`, `invoice_no`, `invoice_date` hoặc cột có unique ratio quá cao có thể được xem là metadata.

Ví dụ với `shopping_behavior_updated.csv`, cột `Customer ID` được phân loại là metadata, còn các cột như `Age`, `Gender`, `Category`, `Purchase Amount (USD)`, `Review Rating`, `Previous Purchases` được dùng làm feature. Người dùng vẫn có thể override vai trò cột trong tab tiền xử lý nếu có lý do nghiệp vụ.

### 2.6. Phân cụm dữ liệu

Clustering là bài toán học không giám sát nhằm chia các điểm dữ liệu thành các nhóm sao cho các điểm trong cùng nhóm giống nhau hơn so với các điểm ở nhóm khác. Khác với classification, clustering không có nhãn đúng sai ban đầu. Trong đề tài này, mục tiêu là phân nhóm khách hàng dựa trên đặc trưng hành vi và thông tin mua sắm.

### 2.7. KMeans

KMeans là thuật toán phân cụm dựa trên tâm cụm. Ban đầu thuật toán chọn K tâm cụm, sau đó gán từng điểm dữ liệu vào tâm gần nhất, cập nhật lại tâm cụm bằng trung bình các điểm trong cụm, rồi lặp lại cho đến khi hội tụ. Trong hệ thống, KMeans được triển khai trong `pipeline/clustering.py` bằng `sklearn.cluster.KMeans`, với `n_init = 10`, `max_iter = 300`, thuật toán mặc định `lloyd` và metric khoảng cách Euclidean.

Trong báo cáo Khai phá dữ liệu, KMeans được xem là thuật toán phân cụm chính. Autoencoder đóng vai trò nâng cấp biểu diễn đặc trưng trước khi KMeans thực hiện gom cụm.

### 2.8. Tự động chọn số cụm K

KMeans cần biết trước số cụm K. Nếu chọn K quá nhỏ, các nhóm khách hàng khác nhau có thể bị gộp chung; nếu chọn K quá lớn, cụm có thể bị chia vụn và khó diễn giải. Hệ thống thử nhiều giá trị K trong khoảng `k_min` đến `k_max`, mặc định 2 đến 10, tính Silhouette Score và Inertia cho từng K. K có Silhouette tốt nhất được ưu tiên, đồng thời có tham khảo elbow khi các điểm số gần nhau.

### 2.9. Các chỉ số đánh giá phân cụm

Silhouette Score đo mức độ một điểm gần với cụm của nó và xa với cụm khác; giá trị càng cao càng tốt. Davies-Bouldin Index đo mức độ cụm compact và tách biệt; giá trị càng thấp càng tốt. Calinski-Harabasz Score đo tỷ lệ phân tán giữa cụm so với trong cụm; giá trị càng cao càng tốt. Inertia là tổng khoảng cách bình phương từ điểm đến tâm cụm, thường dùng để vẽ Elbow.

### 2.10. Cluster profiling

Cluster profiling là bước diễn giải sau khi đã có nhãn cụm. Hệ thống quay lại dữ liệu kết quả có `cluster_label`, sau đó thống kê các đặc trưng gốc theo từng cụm. Với numeric, hệ thống tính mean, median, min, max. Với categorical, hệ thống lấy giá trị phổ biến nhất. Đây là bước diễn giải, không phải bước tạo cụm. Nếu sự khác biệt giữa các cụm nhỏ, báo cáo chỉ nên viết “có xu hướng” thay vì kết luận quá mạnh.

### 2.11. Vai trò của Autoencoder trong báo cáo Khai phá dữ liệu

Trong báo cáo Khai phá dữ liệu, Autoencoder được trình bày như một kỹ thuật hỗ trợ học biểu diễn đặc trưng. Thay vì chạy KMeans trực tiếp trên feature matrix có nhiều chiều sau OneHotEncoder, hệ thống dùng Autoencoder để nén dữ liệu thành latent vector. KMeans sau đó gom cụm trên latent vector. Kết quả test cho thấy với dataset `shopping_behavior_updated.csv`, Silhouette của KMeans trực tiếp khoảng 0,0974, còn Autoencoder + KMeans đạt khoảng 0,3742. Điều này cho thấy biểu diễn latent có thể giúp KMeans hoạt động tốt hơn, nhưng vẫn cần diễn giải trung thực vì chất lượng cụm còn phụ thuộc dataset và feature engineering.

## Chương 3. Phân tích và thiết kế hệ thống Khai phá dữ liệu

### 3.1. Kiến trúc tổng quan hệ thống

Pipeline tổng quan của hệ thống là:

```text
Dữ liệu đầu vào
-> Data profiling
-> EDA
-> Kiểm tra dữ liệu
-> Tiền xử lý
-> Feature matrix
-> Autoencoder
-> Latent vector
-> KMeans
-> Đánh giá
-> Cluster profiling
-> Export
```

Ứng dụng chính là `app.py`, giao diện bằng Streamlit. Các module xử lý nằm trong `pipeline/`: `ingestion.py` đọc dữ liệu; `data_quality.py` profiling, data quality, feature/metadata, preprocessing plan và export; `schema_detection.py` suy luận schema; `preprocessing.py` scale numeric và encode categorical; `embedding.py` xử lý text embedding nếu có; `autoencoder.py` train Autoencoder; `clustering.py` chạy KMeans và chọn K; `evaluation.py` tính metric và t-SNE. Project hiện không có `pipeline/visualization.py`; logic visualization nằm trong `app.py`.

### 3.2. Thiết kế 7 tab giao diện

Tab 1, Tải dữ liệu: cho phép upload CSV/XLSX/JSON hoặc nhập đường dẫn file, sau đó preview dữ liệu và lưu `raw_df`, `current_df`, `data_profile`.

Tab 2, Tổng quan & trực quan hóa dữ liệu: hiển thị số dòng, số cột, missing cells, duplicate rows, profile theo cột và các biểu đồ histogram, boxplot, correlation, scatter, count plot, missing chart.

Tab 3, Kiểm tra & tiền xử lý dữ liệu: phát hiện missing, duplicate, outlier, metadata, high cardinality; cho phép người dùng chọn vai trò cột và xác nhận cleaning plan. Pipeline chính chỉ chạy sau khi `cleaning_confirmed = True`.

Tab 4, Cấu hình mô hình: cấu hình latent dimension, epochs, batch size, learning rate, random seed, auto K hoặc manual K, t-SNE perplexity.

Tab 5, Tiến trình xử lý: chạy `run_pipeline()`, hiển thị trạng thái các stage: kiểm tra dữ liệu, preprocessing, embedding, Autoencoder, latent, chọn K, clustering, evaluation, visualization và export.

Tab 6, Đánh giá kết quả: hiển thị Selected K, Silhouette, Davies-Bouldin, Calinski-Harabasz, số cụm, dòng bị loại; đồng thời hiển thị biểu đồ chọn K, cluster size, Autoencoder loss curve và latent 2D.

Tab 7, Kết quả phân cụm: hiển thị dataframe kèm `cluster_label`, số lượng mỗi cụm, cluster profiling và nút tải CSV.

### 3.3. Module tải dữ liệu

Module tải dữ liệu gồm `load_source()` trong `app.py` và `load_tabular_dataset()` trong `pipeline/ingestion.py`. Hệ thống hỗ trợ CSV, XLSX và JSON. Với CSV, ingestion thử nhiều encoding như UTF-8, UTF-8-SIG, CP1252, Latin-1 và nhiều delimiter. Sau khi đọc, dữ liệu được thêm `original_index` để truy vết dòng gốc.

### 3.4. Module EDA và trực quan hóa

Module EDA nằm chủ yếu trong `render_eda_tab()`. Hệ thống lấy dữ liệu từ `raw_df` và `data_profile`. Nếu dữ liệu lớn, biểu đồ dùng sample tối đa 5.000 dòng để tránh treo ứng dụng. Các biểu đồ EDA giúp người dùng quan sát phân phối, outlier, tương quan, nhóm categorical và missing value trước khi xác nhận tiền xử lý.

### 3.5. Module kiểm tra chất lượng dữ liệu

Module chất lượng dữ liệu nằm trong `pipeline/data_quality.py`. `profile_data()` tạo `DataProfile` gồm số dòng, số cột, tổng missing, duplicate rows, danh sách numeric/categorical/text, feature/metadata, suspicious high-uniqueness columns, summary numeric/categorical. Outlier numeric được phát hiện bằng IQR. Constant và near-constant columns cũng được đánh dấu để có thể loại khỏi dữ liệu.

### 3.6. Module tiền xử lý dữ liệu

`apply_preprocessing_plan()` áp dụng cleaning plan đã xác nhận: drop duplicate, impute missing, cap/drop outlier, drop constant columns. Sau đó `preprocess_features()` trong `pipeline/preprocessing.py` tạo feature matrix dạng số bằng scaling numeric và encoding categorical. Text tự do nếu có có thể đi qua `embed_text_features()`.

### 3.7. Module phân cụm

Module phân cụm nằm trong `pipeline/clustering.py`. KMeans nhận đầu vào là `Z_latent` từ Encoder, không nhận dữ liệu gốc. Hệ thống thử các K hợp lệ, tính silhouette và inertia, chọn K tốt, sau đó fit KMeans cuối cùng để tạo `cluster_labels`.

### 3.8. Module đánh giá và visualization kết quả

Module đánh giá nằm trong `pipeline/evaluation.py`, tính Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score và chuẩn bị t-SNE 2D nếu bật `compute_tsne`. t-SNE chỉ phục vụ quan sát latent space, không tạo nhãn cụm và không dùng để train.

### 3.9. Module export/report

Các hàm export nằm trong `pipeline/data_quality.py`: `export_removed_rows()`, `export_cleaning_report()`, `export_cluster_results()`, `export_final_report()`. Output chính gồm `outputs/removed_rows/removed_rows_<timestamp>.csv`, `outputs/reports/data_cleaning_report_<timestamp>.json`, `outputs/clustering_results/customer_segments_<timestamp>.csv` và `outputs/reports/final_report_<timestamp>.json`.

## Chương 4. Thực nghiệm và kết quả Khai phá dữ liệu

### 4.1. Dataset sử dụng

Dataset ưu tiên là `DataSet/shopping_behavior_updated.csv`, gồm 3.900 dòng và 18 cột. Cột metadata là `Customer ID`. Các cột feature gồm `Age`, `Gender`, `Item Purchased`, `Category`, `Purchase Amount (USD)`, `Location`, `Size`, `Color`, `Season`, `Review Rating`, `Subscription Status`, `Shipping Type`, `Discount Applied`, `Promo Code Used`, `Previous Purchases`, `Payment Method`, `Frequency of Purchases`.

Dataset so sánh là `DataSet/customer_shopping_data.csv`, gồm 99.457 dòng và 10 cột; test report sử dụng sample 5.000 dòng. Metadata gồm `invoice_no`, `customer_id`, `invoice_date`; feature gồm `gender`, `age`, `category`, `quantity`, `price`, `payment_method`, `shopping_mall`.

### 4.2. Kết quả data profiling

Với `shopping_behavior_updated.csv`, report ghi nhận không có missing cells, không có duplicate rows, có 5 cột numeric nếu tính cả `Customer ID`, 13 cột categorical, không có text columns. Sau khi phân loại vai trò cột, `Customer ID` là metadata và không đi vào feature matrix.

Với `customer_shopping_data.csv`, report ghi nhận sample 5.000 dòng không có missing và duplicate. Các cột numeric gồm `age`, `quantity`, `price`; categorical gồm `gender`, `category`, `payment_method`, `invoice_date`, `shopping_mall`; `invoice_no` và `customer_id` được suy luận là text/high uniqueness nhưng được đưa vào metadata, không dùng train.

### 4.3. Kết quả trực quan hóa dữ liệu

[Chèn hình histogram tại đây]  
Gợi ý: histogram của `Age` hoặc `Purchase Amount (USD)` để minh họa phân phối biến số.

[Chèn hình boxplot tại đây]  
Gợi ý: boxplot của `Purchase Amount (USD)` hoặc `price` để minh họa outlier.

[Chèn hình heatmap tại đây]  
Gợi ý: correlation heatmap giữa các cột numeric.

[Chèn hình scatterplot tại đây]  
Gợi ý: scatter giữa `Age` và `Purchase Amount (USD)`.

[Chèn hình count plot tại đây]  
Gợi ý: count plot của `Category`, `Payment Method` hoặc `Frequency of Purchases`.

[Chèn hình missing chart tại đây]  
Gợi ý: nếu dataset không có missing, có thể chèn màn hình hệ thống báo dữ liệu không có missing value.

### 4.4. Kết quả tiền xử lý

Với `shopping_behavior_updated.csv`, `Customer ID` bị loại khỏi feature matrix vì là metadata. Numeric được scale, categorical được OneHotEncoder, không có text embedding. Feature matrix có shape `(3900, 143)`. Cleaning report ghi nhận 3.900 dòng được giữ lại, không có removed rows.

Với `customer_shopping_data.csv`, sample 5.000 dòng tạo feature matrix `(5000, 26)`. Cột `price` có 262 giá trị outlier được cap bằng IQR theo report. Metadata không đi vào feature matrix nhưng vẫn được giữ để đối chiếu trong kết quả.

### 4.5. Kết quả tự động chọn K

Với `shopping_behavior_updated.csv`, hệ thống thử các K từ 2 đến 8 trong report test. KMeans trực tiếp trên feature preprocessing có best K = 2 với Silhouette khoảng 0,0974. KMeans trên latent vector chọn K = 4, Silhouette khoảng 0,3742. Trong báo cáo nên chèn biểu đồ Silhouette theo K và Elbow/Inertia để minh họa quá trình chọn K.

### 4.6. Kết quả phân cụm

Với `shopping_behavior_updated.csv`, kết quả tốt nhất trong report là Autoencoder + KMeans trên latent vector, selected K khoảng 4. Latent shape là `(3900, 8)`, nghĩa là mỗi khách hàng được biểu diễn bằng vector tiềm ẩn 8 chiều trước khi gom cụm. Bảng kết quả phân cụm được xuất ra `outputs/clustering_results/customer_segments_20260603_153128.csv`.

### 4.7. Diễn giải cụm

Diễn giải cụm phải dựa trên đặc trưng gốc, không diễn giải từng chiều latent như `z1`, `z2` là Age hay Purchase Amount. Sau khi có `cluster_label`, hệ thống thống kê theo cụm trên các cột như `Purchase Amount (USD)`, `Previous Purchases`, `Review Rating`, `Category`, `Payment Method`. Nếu một cụm có trung bình `Purchase Amount (USD)` cao hơn các cụm khác, có thể viết “cụm này có xu hướng chi tiêu cao hơn”. Nếu `Previous Purchases` cao hơn, có thể diễn giải là nhóm có xu hướng mua lặp lại nhiều hơn. Nếu chênh lệch nhỏ, không nên kết luận quá mạnh.

### 4.8. Nhận xét kết quả Khai phá dữ liệu

Hệ thống đã hoàn thành quy trình khai phá dữ liệu từ tải dữ liệu, EDA, kiểm tra chất lượng, tiền xử lý, tạo feature matrix, phân cụm, đánh giá và export. Kết quả thực nghiệm cho thấy Autoencoder giúp cải thiện biểu diễn đặc trưng trước khi chạy KMeans, thể hiện qua Silhouette tăng từ khoảng 0,0974 lên khoảng 0,3742 trên `shopping_behavior_updated.csv`. Tuy nhiên, kết quả không nên được mô tả là hoàn hảo; chất lượng cụm còn phụ thuộc dữ liệu, đặc trưng đầu vào, cách mã hóa categorical và cách diễn giải nghiệp vụ.

## Chương 5. Kết luận và hướng phát triển

### 5.1. Kết luận

Đề tài đã xây dựng được một hệ thống Streamlit hỗ trợ phân cụm khách hàng theo quy trình khai phá dữ liệu. Hệ thống có đủ các bước EDA, data profiling, preprocessing có xác nhận, feature/metadata selection, KMeans, tự động chọn K, evaluation, cluster profiling và export. Với bài toán customer segmentation không có nhãn, hệ thống phù hợp để hỗ trợ phân tích nhóm khách hàng ban đầu.

### 5.2. Hạn chế

Kết quả phân cụm phụ thuộc mạnh vào dataset và feature engineering. Một số cụm có thể chưa tách biệt mạnh nếu dữ liệu không chứa đặc trưng hành vi đủ rõ. Dataset dạng giao dịch chưa có RFM hoặc feature thời gian sâu. Latent vector khó diễn giải trực tiếp, nên bắt buộc phải quay lại đặc trưng gốc để profiling. Hệ thống hiện chưa có explainability nâng cao và chưa triển khai production.

### 5.3. Hướng phát triển

Hướng phát triển gồm thêm RFM, thêm feature engineering theo thời gian, thêm metric DBI/CH theo từng K trong biểu đồ chọn K, thử các thuật toán clustering khác như DBSCAN, Gaussian Mixture hoặc Hierarchical Clustering, thêm chức năng gán cụm cho khách hàng mới và cải thiện phần diễn giải cụm bằng feature importance hoặc thống kê khác biệt giữa cụm.

---

# II. Nội dung báo cáo Deep Learning

## Chương 1. Tổng quan đề tài

### 1.1. Lý do chọn đề tài

Dữ liệu khách hàng dạng bảng thường có nhiều chiều và nhiều kiểu dữ liệu. Sau tiền xử lý, các biến categorical có thể được mã hóa thành nhiều cột OneHot, làm feature matrix có số chiều cao hơn nhiều so với số cột gốc. KMeans trực tiếp trên ma trận này có thể bị ảnh hưởng bởi nhiễu, dữ liệu thưa và khoảng cách Euclidean trong không gian nhiều chiều.

Deep Learning có khả năng học biểu diễn đặc trưng tự động. Trong bài toán không có nhãn, Autoencoder là lựa chọn phù hợp vì mô hình học cách nén dữ liệu vào latent vector rồi tái tạo lại dữ liệu đầu vào. Khi latent vector giữ được thông tin quan trọng và giảm bớt nhiễu, KMeans có thể phân cụm tốt hơn trên latent space. Vì vậy, đề tài kết hợp Autoencoder và KMeans theo hướng Deep Clustering.

### 1.2. Mục tiêu đề tài

Mục tiêu Deep Learning là xây dựng hệ thống phân cụm khách hàng dựa trên Autoencoder. Hệ thống tạo feature matrix sau preprocessing, train Autoencoder để học latent representation, dùng Encoder trích xuất `Z_latent`, chạy KMeans trên latent vector, đánh giá clustering và so sánh với KMeans trực tiếp trên feature matrix.

### 1.3. Phạm vi đề tài

Hệ thống xử lý dữ liệu dạng bảng, dùng fully connected Autoencoder. Project không dùng CNN vì dữ liệu không phải ảnh, không dùng supervised learning vì không có nhãn, chưa triển khai Deep Embedded Clustering hoàn chỉnh, chưa triển khai production.

### 1.4. Đóng góp của hệ thống

Hệ thống đóng góp một pipeline Deep Clustering có UI: tiền xử lý dữ liệu, tạo feature matrix, train Autoencoder, trích xuất latent vector, tự động chọn K, clustering bằng KMeans, đánh giá metric, trực quan hóa latent space bằng t-SNE và export kết quả.

## Chương 2. Cơ sở lý thuyết Deep Learning

### 2.1. Deep Learning và Artificial Neural Network

Deep Learning là nhánh của Machine Learning sử dụng mạng neural nhiều lớp để học biểu diễn dữ liệu. Với dữ liệu dạng bảng, fully connected layers thường được sử dụng vì mỗi dòng dữ liệu được biểu diễn bằng một vector đặc trưng. Mạng học thông qua forward pass, tính loss, backpropagation và cập nhật trọng số bằng optimizer như Adam.

### 2.2. Học không giám sát

Học không giám sát là tình huống dữ liệu không có nhãn `y`. Mô hình không học để dự đoán nhãn đúng, mà học cấu trúc ẩn hoặc biểu diễn của dữ liệu. Autoencoder trong đề tài là mô hình không giám sát vì input và target đều là dữ liệu đầu vào sau tiền xử lý.

### 2.3. Autoencoder

Autoencoder gồm Encoder, latent vector và Decoder. Encoder nén input thành biểu diễn thấp chiều hơn. Decoder tái tạo lại input từ latent vector. Mục tiêu train là giảm reconstruction loss, trong project là MSELoss.

```text
X_preprocessed
-> Encoder
-> Z_latent
-> Decoder
-> X_reconstructed
```

Khi huấn luyện Autoencoder, input và target đều là ma trận đặc trưng sau tiền xử lý. Mục tiêu của mô hình là tái tạo lại chính dữ liệu đầu vào với sai số thấp nhất.

### 2.4. Cấu trúc Autoencoder trong hệ thống

Autoencoder nằm trong `pipeline/autoencoder.py`. Cấu hình chính gồm `epochs`, `batch_size`, `learning_rate`, `latent_dim`, `min_latent`, `max_latent`, `hidden_dims`, `max_hidden_layers`, `activation`, `loss`, `optimizer`, `use_gpu`, `random_seed`. Mặc định `epochs = 50`, `batch_size = 128`, `learning_rate = 0.001`, activation `relu`, loss `mse`, optimizer `adam`, random seed 42. Trong test report, epochs dùng là 6.

Số lớp không cố định hoàn toàn. Nếu `hidden_dims` không được truyền vào, hệ thống tự chọn hidden dimensions dựa trên `input_dim`, `latent_dim` và `max_hidden_layers`. Encoder gồm các Linear layer giảm dần về latent dimension. Decoder đối xứng, đi từ latent dimension trở lại input dimension. Đây là fully connected Autoencoder.

### 2.5. Dữ liệu truyền vào Autoencoder

Autoencoder không nhận dữ liệu gốc trực tiếp. Dữ liệu đầu vào là feature matrix dạng số sau tiền xử lý. Metadata như `Customer ID`, `invoice_no`, `customer_id`, `invoice_date` không đưa vào train. Numeric được scale, categorical được encode, text tự do nếu có mới dùng embedding. Trong project, categorical ngắn như `Gender`, `Category`, `Payment Method` được xử lý bằng OneHotEncoder hoặc OrdinalEncoder, không phải text embedding.

### 2.6. Latent representation

Latent vector là biểu diễn tiềm ẩn do Encoder học được. Mỗi chiều latent không tương ứng trực tiếp với một cột gốc. Ví dụ không thể nói `z1` là Age hoặc `z2` là Purchase Amount. Latent vector là tổ hợp học được từ nhiều đặc trưng sau preprocessing. Hệ thống dùng latent vector để gom cụm, sau đó diễn giải cụm bằng đặc trưng gốc.

### 2.7. Deep Clustering

Deep Clustering trong đề tài là sự kết hợp giữa deep representation learning và clustering. Autoencoder học biểu diễn `Z_latent`; KMeans gom cụm trên `Z_latent`. Cách này khác với KMeans trực tiếp trên feature matrix sau preprocessing. Autoencoder có thể giảm chiều, giảm nhiễu và tạo không gian biểu diễn phù hợp hơn cho clustering.

### 2.8. KMeans trên latent vector

Sau khi train, hệ thống dùng Encoder để biến mỗi dòng `X_preprocessed` thành một vector latent. KMeans nhận `Z_latent` làm đầu vào và gán cụm dựa trên khoảng cách đến tâm cụm trong latent space. Decoder không dùng để clustering; Decoder chỉ phục vụ quá trình huấn luyện tái tạo.

### 2.9. t-SNE

t-SNE chỉ dùng để trực quan hóa. t-SNE không phải thuật toán phân cụm chính, không dùng để train, không tạo cluster label. Trong project, t-SNE chiếu latent vector nhiều chiều xuống 2D để người dùng nhìn phân bố cụm. Có thể ghi nhớ câu: **t-SNE để nhìn, metric để đánh giá.**

### 2.10. Các chỉ số đánh giá

Báo cáo Deep Learning dùng hai nhóm chỉ số. Nhóm thứ nhất là reconstruction loss của Autoencoder, phản ánh khả năng tái tạo input. Loss giảm cho thấy mô hình học được biểu diễn có ích, nhưng loss thấp không tự động đảm bảo cụm tốt. Nhóm thứ hai là clustering metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz. Ngoài ra cần so sánh KMeans trực tiếp với Autoencoder + KMeans để chứng minh vai trò của học biểu diễn.

## Chương 3. Phân tích và thiết kế hệ thống Deep Learning

### 3.1. Kiến trúc tổng quan

```text
Dữ liệu
-> preprocessing
-> feature matrix
-> Autoencoder
-> latent vector
-> KMeans
-> evaluation
-> visualization
```

### 3.2. Module tiền xử lý dữ liệu

Tiền xử lý quan trọng ngang với Autoencoder. Nếu dữ liệu đầu vào sai, chứa ID, missing chưa xử lý hoặc categorical chưa encode, Autoencoder sẽ học biểu diễn sai. Trong project, `feature_df = current_df[feature_columns]`, nghĩa là chỉ các cột feature đã xác nhận mới đi vào pipeline. `preprocess_features()` tạo `X_numeric`; `embed_text_features()` tạo `X_embedding` nếu có text; Autoencoder nối các nguồn feature hợp lệ thành `X_final`.

### 3.3. Module Autoencoder

`train_autoencoder_features()` gọi `AutoencoderModule.run()`. Module chuẩn bị ma trận train, chọn kiến trúc, train mô hình PyTorch, tính loss history và trích xuất latent. Model dùng `nn.Linear`, activation ReLU mặc định, `nn.MSELoss` và `torch.optim.Adam`.

### 3.4. Module trích xuất latent vector

Sau khi train, `_encode_latent()` chạy `model.encode()` trên toàn bộ feature matrix. Kết quả là `Z_latent` có shape `(n_samples, latent_dim)`. Với `shopping_behavior_updated.csv`, latent shape là `(3900, 8)`. Với `customer_shopping_data.csv` sample 5.000 dòng, latent shape là `(5000, 6)`.

### 3.5. Module clustering

Clustering nằm trong `pipeline/clustering.py`. KMeans chạy trên `Z_latent`, thử các K từ `k_min` đến `k_max`, chọn K dựa trên Silhouette và có tham khảo Elbow/Inertia. Baseline KMeans trực tiếp trên feature matrix không nằm trong UI chính, nhưng có trong report đánh giá dataset.

### 3.6. Module evaluation

Evaluation tính reconstruction loss từ metadata Autoencoder và clustering metrics từ `evaluate_clustering()`. Các metric được hiển thị trong tab Đánh giá kết quả. Với `shopping_behavior_updated.csv`, Autoencoder loss giảm từ 0,1088 xuống 0,0704; latent KMeans đạt Silhouette khoảng 0,3742.

### 3.7. Module visualization

Visualization gồm Autoencoder loss curve, biểu đồ chọn K, cluster size và latent space t-SNE. Loss curve giúp quan sát quá trình train. Latent t-SNE giúp nhìn phân bố cụm nhưng không thay thế metric.

## Chương 4. Thực nghiệm và đánh giá Deep Learning

### 4.1. Dataset thực nghiệm

Dataset chính `shopping_behavior_updated.csv` có 3.900 dòng, 18 cột, metadata `Customer ID`, 17 feature columns. Dataset so sánh `customer_shopping_data.csv` có 99.457 dòng, test sample 5.000 dòng, 10 cột, metadata `invoice_no`, `customer_id`, `invoice_date`.

### 4.2. Cấu hình tiền xử lý

Với `shopping_behavior_updated.csv`, feature matrix sau preprocessing có shape `(3900, 143)`. Numeric được scale, categorical được OneHotEncoder, không có text embedding. Với `customer_shopping_data.csv`, feature matrix có shape `(5000, 26)`; `price` có outlier được cap IQR.

### 4.3. Cấu hình Autoencoder

Với `shopping_behavior_updated.csv`, `input_dim = 143`, `latent_dim = 8`, tức nén từ 143 chiều xuống 8 chiều. Với `customer_shopping_data.csv`, `input_dim = 26`, `latent_dim = 6`, tức nén từ 26 chiều xuống 6 chiều. Cấu trúc là fully connected Autoencoder, Encoder/Decoder đối xứng, loss MSE, optimizer Adam, random seed 42. Thiết bị train là CPU hoặc GPU tùy `torch.cuda.is_available()` và cấu hình `use_gpu`.

### 4.4. Kết quả huấn luyện Autoencoder

Với `shopping_behavior_updated.csv`, loss giảm từ 0,1088 xuống 0,0704 sau 6 epochs theo report. Với `customer_shopping_data.csv`, loss giảm từ 0,2574 xuống 0,1612. Loss giảm cho thấy Autoencoder học được cách tái tạo feature matrix tốt hơn qua các epoch, nhưng cần kết hợp clustering metrics để đánh giá hiệu quả phân cụm.

[Chèn biểu đồ Autoencoder loss curve tại đây]

### 4.5. Kết quả latent representation

Với `shopping_behavior_updated.csv`, latent vector có shape `(3900, 8)`. Với `customer_shopping_data.csv`, latent vector có shape `(5000, 6)`. Các vector này là đầu vào cho KMeans. Không diễn giải từng chiều latent riêng lẻ; chỉ dùng latent để gom cụm, rồi quay lại đặc trưng gốc để profiling.

[Chèn biểu đồ latent space t-SNE tại đây]

### 4.6. Kết quả clustering

Với `shopping_behavior_updated.csv`, Autoencoder + KMeans chọn K khoảng 4, Silhouette khoảng 0,3742, Davies-Bouldin khoảng 0,9278, Calinski-Harabasz khoảng 2936,77. Với `customer_shopping_data.csv` sample 5.000 dòng, latent KMeans chọn K khoảng 3, Silhouette khoảng 0,4625, Davies-Bouldin khoảng 0,7468, Calinski-Harabasz khoảng 8099,31.

### 4.7. So sánh KMeans trực tiếp và Autoencoder + KMeans

| Phương pháp | Dữ liệu đầu vào | K | Silhouette | Davies-Bouldin | Calinski-Harabasz | Nhận xét |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| KMeans trực tiếp | Feature matrix sau preprocessing của `shopping_behavior_updated.csv` | 2 | 0,0974 | 2,9653 | 432,11 | Baseline thấp |
| Autoencoder + KMeans | Latent vector của `shopping_behavior_updated.csv` | 4 | 0,3742 | 0,9278 | 2936,77 | Cải thiện rõ theo metric |
| KMeans trực tiếp | Feature matrix sau preprocessing của `customer_shopping_data.csv` sample 5.000 dòng | 2 | 0,1956 | 1,8544 | 1004,64 | Baseline |
| Autoencoder + KMeans | Latent vector của `customer_shopping_data.csv` sample 5.000 dòng | 3 | 0,4625 | 0,7468 | 8099,31 | Cải thiện rõ theo metric |

Kết quả cho thấy Autoencoder giúp cải thiện chất lượng phân cụm so với KMeans trực tiếp trên feature matrix trong các test đã ghi nhận. Silhouette tăng, Davies-Bouldin giảm và Calinski-Harabasz tăng. Tuy nhiên, kết quả cần được trình bày trung thực: metric tốt hơn baseline không có nghĩa mọi cụm đều có ý nghĩa kinh doanh mạnh; vẫn cần cluster profiling để diễn giải.

### 4.8. Phân tích kết quả

Autoencoder giúp giảm chiều và giảm nhiễu trong không gian đặc trưng sau preprocessing. Với dữ liệu có nhiều categorical được OneHot, latent vector có thể gom thông tin quan trọng vào số chiều nhỏ hơn, giúp KMeans tìm cụm rõ hơn. Tuy vậy, latent vector khó giải thích trực tiếp, nên hệ thống không diễn giải `z1`, `z2` như cột gốc. Thay vào đó, hệ thống gắn `cluster_label` trở lại dữ liệu và thống kê các đặc trưng gốc.

### 4.9. Trả lời câu hỏi “gom cụm dựa trên đặc trưng nào?”

Hệ thống gom cụm dựa trên latent features, tức các đặc trưng tiềm ẩn do Autoencoder học từ feature matrix sau preprocessing. Latent features không phải là cột gốc đơn lẻ, mà là tổ hợp thông tin từ nhiều cột như tuổi, số tiền mua hàng, danh mục sản phẩm, phương thức thanh toán và tần suất mua hàng sau khi được scale/encode. **Gom cụm bằng latent vector, diễn giải cụm bằng đặc trưng gốc.**

## Chương 5. Kết luận và hướng phát triển

### 5.1. Kết luận

Đề tài đã xây dựng được hệ thống Deep Clustering cho dữ liệu khách hàng dạng bảng. Autoencoder học latent representation từ feature matrix sau preprocessing; KMeans chạy trên latent vector; kết quả đánh giá cho thấy cải thiện so với baseline KMeans trực tiếp. Hệ thống có UI, EDA, preprocessing, evaluation, visualization và export.

### 5.2. Hạn chế

Latent vector khó diễn giải trực tiếp. Kết quả phụ thuộc dataset, feature engineering và cấu hình mô hình. Project hiện chưa triển khai DEC thật sự, chưa có chức năng gán cụm khách hàng mới trong UI nếu chưa bổ sung, chưa có explainability nâng cao và chưa có pipeline production.

### 5.3. Hướng phát triển

Có thể phát triển thêm Deep Embedded Clustering, Variational Autoencoder, chức năng predict cluster cho khách hàng mới, RFM/time features, SHAP hoặc permutation importance để giải thích cụm và thử thêm các mô hình clustering khác.

---

# III. Các đoạn giải thích quan trọng

## 1. Giải thích hệ thống tổng quan

Hệ thống là một ứng dụng Streamlit hỗ trợ phân cụm khách hàng tự động. Người dùng đưa dữ liệu khách hàng vào hệ thống, sau đó hệ thống thực hiện profiling, EDA, kiểm tra chất lượng dữ liệu, tiền xử lý, tạo feature matrix, train Autoencoder, trích xuất latent vector, chọn số cụm K, phân cụm bằng KMeans, đánh giá metric, trực quan hóa và xuất kết quả. Điểm quan trọng của hệ thống là không đưa dữ liệu thô trực tiếp vào mô hình, mà có bước xác nhận preprocessing và lựa chọn feature/metadata trước khi chạy pipeline chính.

## 2. Vì sao cần tiền xử lý dữ liệu

Dữ liệu khách hàng thường có nhiều kiểu dữ liệu khác nhau, gồm số, nhóm phân loại, ngày tháng, mã định danh và đôi khi có text. Autoencoder và KMeans đều cần đầu vào dạng số, không chứa missing bất thường và không bị ảnh hưởng quá mạnh bởi scale khác nhau giữa các cột. Vì vậy numeric cần được scale, categorical cần được encode, missing value cần được xử lý, outlier cần được xem xét và metadata cần được loại khỏi tập train.

## 3. Vì sao Customer ID không dùng để train

`Customer ID` là mã định danh, không mô tả hành vi hay đặc điểm tiêu dùng. Nếu đưa `Customer ID` vào train, mô hình có thể học khoảng cách giả giữa các mã khách hàng, làm sai lệch kết quả phân cụm. Vì vậy hệ thống giữ `Customer ID` làm metadata để đối chiếu kết quả, nhưng không đưa vào Autoencoder hoặc KMeans.

## 4. Categorical dạng chữ xử lý bằng OneHotEncoder, không phải text embedding

Các cột như `Gender`, `Category`, `Payment Method`, `Season` là categorical ngắn, mỗi giá trị là một nhãn rời rạc. Chúng phù hợp với OneHotEncoder hoặc OrdinalEncoder. Text embedding chỉ phù hợp với văn bản tự do có ý nghĩa ngữ nghĩa như mô tả dài hoặc câu. Trong dataset chính hiện tại không có text tự do, nên categorical được encode chứ không dùng embedding.

## 5. Autoencoder nhận gì vào

Autoencoder nhận feature matrix dạng số sau tiền xử lý. Feature matrix này được tạo từ numeric đã scale, categorical đã encode và text embedding nếu có text tự do. Input và target khi train đều là ma trận này. Mô hình học cách tái tạo lại chính input, từ đó buộc Encoder học biểu diễn nén trong latent space.

## 6. Autoencoder có mấy phần, mấy lớp

Autoencoder có hai phần chính: Encoder và Decoder. Encoder nén input thành latent vector; Decoder tái tạo input từ latent vector. Số lớp hidden không cố định hoàn toàn mà được hệ thống chọn dựa trên `input_dim`, `latent_dim` và `max_hidden_layers`, hoặc lấy từ `hidden_dims` nếu người dùng cấu hình. Trong code, mỗi lớp chính là `nn.Linear` kết hợp activation như ReLU.

## 7. Latent vector

Latent vector là biểu diễn tiềm ẩn sau Encoder. Mỗi dòng khách hàng được biến thành một vector ngắn hơn, ví dụ từ 143 chiều xuống 8 chiều. Các chiều latent không có nghĩa trực tiếp như Age hoặc Purchase Amount, mà là tổ hợp thông tin học được từ nhiều đặc trưng.

## 8. Vì sao dùng KMeans trên latent vector

KMeans nhạy với không gian đặc trưng và khoảng cách. Nếu chạy trên feature matrix nhiều chiều sau OneHot, khoảng cách có thể bị nhiễu. Latent vector giúp nén thông tin và có thể tạo không gian compact hơn. Vì vậy KMeans chạy trên latent vector thường cho kết quả tốt hơn baseline trong test của project.

## 9. t-SNE

t-SNE là kỹ thuật giảm chiều để trực quan hóa. Trong hệ thống, t-SNE chiếu latent vector xuống 2D để người dùng nhìn các cụm trên biểu đồ. t-SNE không tạo cluster label, không được dùng để train và không phải thuật toán phân cụm chính.

## 10. Cluster profiling

Cluster profiling là bước quay lại dữ liệu gốc sau khi đã có nhãn cụm. Hệ thống thống kê mean, median, min, max của numeric và giá trị phổ biến của categorical theo từng cluster. Nhờ đó người dùng diễn giải cụm bằng các đặc trưng quen thuộc như Purchase Amount, Previous Purchases, Review Rating, thay vì diễn giải trực tiếp các chiều latent.

## 11. Vì sao hai báo cáo có trọng tâm khác nhau

Cùng một hệ thống có thể phục vụ hai môn, nhưng trọng tâm khác nhau. Với Khai phá dữ liệu, trọng tâm là quy trình phát hiện tri thức từ dữ liệu: EDA, chất lượng dữ liệu, preprocessing, clustering, đánh giá và cluster profiling. Với Deep Learning, trọng tâm là cách Autoencoder học representation, reconstruction loss, latent vector và việc chứng minh Autoencoder + KMeans cải thiện so với KMeans trực tiếp.

---

# IV. Danh sách hình cần chèn

## Báo cáo Khai phá dữ liệu

| Tên hình | Chương | Minh họa | Caption gợi ý |
| --- | --- | --- | --- |
| Giao diện tải dữ liệu | Chương 3 | Tab upload/nhập đường dẫn và preview dataset | Hình 3.1. Giao diện tải dữ liệu khách hàng |
| Tab EDA | Chương 3 | Tổng quan dữ liệu và profile theo cột | Hình 3.2. Giao diện tổng quan và trực quan hóa dữ liệu |
| Histogram | Chương 4 | Phân phối biến numeric | Hình 4.1. Histogram phân phối dữ liệu khách hàng |
| Boxplot | Chương 4 | Outlier và phân bố numeric | Hình 4.2. Boxplot phát hiện giá trị bất thường |
| Correlation heatmap | Chương 4 | Tương quan giữa các biến numeric | Hình 4.3. Bản đồ tương quan các đặc trưng số |
| Missing chart | Chương 4 | Tình trạng missing value | Hình 4.4. Biểu đồ giá trị thiếu theo cột |
| Tab tiền xử lý | Chương 3 | Cleaning plan, feature/metadata | Hình 3.3. Giao diện kiểm tra và tiền xử lý dữ liệu |
| Biểu đồ chọn K | Chương 4 | Silhouette và Elbow/Inertia | Hình 4.5. Biểu đồ hỗ trợ chọn số cụm K |
| Cluster size | Chương 4 | Số khách hàng mỗi cụm | Hình 4.6. Phân bố kích thước các cụm |
| Cluster profiling | Chương 4 | Thống kê đặc trưng gốc theo cụm | Hình 4.7. Bảng mô tả đặc điểm từng cụm |
| File export/report | Chương 3 hoặc 4 | Các file output sinh ra | Hình 4.8. Kết quả export của hệ thống |

## Báo cáo Deep Learning

| Tên hình | Chương | Minh họa | Caption gợi ý |
| --- | --- | --- | --- |
| Sơ đồ pipeline Deep Clustering | Chương 3 | Preprocessing -> Autoencoder -> KMeans | Hình 3.1. Pipeline Deep Clustering trong hệ thống |
| Sơ đồ Autoencoder | Chương 2 | Encoder, latent, Decoder | Hình 2.1. Cấu trúc Autoencoder |
| Cấu hình Autoencoder trên UI | Chương 3 | latent_dim, epochs, batch_size, learning_rate | Hình 3.2. Giao diện cấu hình Autoencoder |
| Loss curve | Chương 4 | Reconstruction loss qua epoch | Hình 4.1. Đường loss huấn luyện Autoencoder |
| Latent space t-SNE | Chương 4 | Chiếu latent vector xuống 2D | Hình 4.2. Trực quan hóa latent space bằng t-SNE |
| Bảng so sánh baseline | Chương 4 | KMeans trực tiếp và Autoencoder + KMeans | Hình 4.3. So sánh kết quả clustering |
| Cluster result | Chương 4 | Bảng khách hàng kèm cluster_label | Hình 4.4. Kết quả phân cụm khách hàng |
| Final report | Chương 4 | JSON report chứa cấu hình và metric | Hình 4.5. Báo cáo kết quả cuối cùng |

---

# V. Danh sách bảng cần có

## 1. Bảng mô tả dataset

| Dataset | Số dòng | Số cột | Ghi chú |
| --- | ---: | ---: | --- |
| `shopping_behavior_updated.csv` | 3.900 | 18 | Dataset chính |
| `customer_shopping_data.csv` | 99.457, test sample 5.000 | 10 | Dataset so sánh |

Ghi chú: số liệu lấy từ project và report test.

## 2. Bảng phân loại feature/metadata

| Dataset | Feature columns | Metadata columns |
| --- | --- | --- |
| `shopping_behavior_updated.csv` | Age, Gender, Category, Purchase Amount, Review Rating, Previous Purchases, ... | Customer ID |
| `customer_shopping_data.csv` | gender, age, category, quantity, price, payment_method, shopping_mall | invoice_no, customer_id, invoice_date |

Ghi chú: metadata được giữ để đối chiếu, không dùng train.

## 3. Bảng cấu hình tiền xử lý

| Thành phần | Cách xử lý |
| --- | --- |
| Missing numeric | mean/median/drop rows tùy plan |
| Missing categorical | mode/unknown |
| Duplicate | drop |
| Outlier | keep/cap IQR/drop IQR/drop Z-score |
| Numeric | StandardScaler/MinMaxScaler/RobustScaler |
| Categorical | OneHotEncoder/OrdinalEncoder |
| Text | embedding nếu là text tự do |

## 4. Bảng cấu hình Autoencoder

| Tham số | Giá trị trong project |
| --- | --- |
| Kiểu mô hình | Fully connected Autoencoder |
| Loss | MSELoss |
| Optimizer | Adam |
| Activation | ReLU mặc định |
| Epochs mặc định | 50 |
| Epochs test | 6 |
| Batch size mặc định | 128 |
| Learning rate | 0,001 |
| Random seed | 42 |

## 5. Bảng kết quả chọn K

| Dataset | Phương pháp | K được chọn | Metric chính |
| --- | --- | ---: | --- |
| `shopping_behavior_updated.csv` | KMeans trực tiếp | 2 | Silhouette 0,0974 |
| `shopping_behavior_updated.csv` | Autoencoder + KMeans | 4 | Silhouette 0,3742 |
| `customer_shopping_data.csv` | KMeans trực tiếp | 2 | Silhouette 0,1956 |
| `customer_shopping_data.csv` | Autoencoder + KMeans | 3 | Silhouette 0,4625 |

## 6. Bảng metric clustering

| Dataset | Phương pháp | Silhouette | Davies-Bouldin | Calinski-Harabasz |
| --- | --- | ---: | ---: | ---: |
| shopping | KMeans trực tiếp | 0,0974 | 2,9653 | 432,11 |
| shopping | Autoencoder + KMeans | 0,3742 | 0,9278 | 2936,77 |
| customer shopping | KMeans trực tiếp | 0,1956 | 1,8544 | 1004,64 |
| customer shopping | Autoencoder + KMeans | 0,4625 | 0,7468 | 8099,31 |

## 7. Bảng so sánh baseline

Dùng lại bảng ở mục 4.7 báo cáo Deep Learning.

## 8. Bảng cluster profiling

| Cluster | Số dòng | Purchase Amount mean | Previous Purchases mean | Review Rating mean | Category phổ biến |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | cần lấy từ output CSV | cần tính từ output CSV | cần tính từ output CSV | cần tính từ output CSV | cần tính từ output CSV |
| 1 | cần lấy từ output CSV | cần tính từ output CSV | cần tính từ output CSV | cần tính từ output CSV | cần tính từ output CSV |

Ghi chú: cần chụp hoặc export bảng cluster profiling từ tab Kết quả phân cụm để điền số cụ thể.

## 9. Bảng output files

| File | Ý nghĩa |
| --- | --- |
| `outputs/removed_rows/removed_rows_<timestamp>.csv` | Dòng bị loại khi cleaning |
| `outputs/reports/data_cleaning_report_<timestamp>.json` | Báo cáo tiền xử lý |
| `outputs/clustering_results/customer_segments_<timestamp>.csv` | Dữ liệu kèm nhãn cụm |
| `outputs/reports/final_report_<timestamp>.json` | Báo cáo cuối pipeline |

---

# VI. Câu hỏi và trả lời khi bảo vệ

1. Vì sao đề tài này thuộc Khai phá dữ liệu?  
Vì hệ thống thực hiện quy trình phát hiện tri thức từ dữ liệu khách hàng không có nhãn: EDA, kiểm tra chất lượng dữ liệu, tiền xử lý, phân cụm, đánh giá và diễn giải cụm.

2. Vì sao đề tài này thuộc Deep Learning?  
Vì hệ thống dùng Autoencoder, một mạng neural, để học latent representation từ feature matrix trước khi phân cụm.

3. Autoencoder nhận gì vào?  
Autoencoder nhận ma trận đặc trưng dạng số sau tiền xử lý, gồm numeric đã scale, categorical đã encode và text embedding nếu có.

4. Autoencoder có mấy lớp?  
Autoencoder gồm Encoder và Decoder. Số hidden layer không cố định hoàn toàn, được chọn theo input_dim, latent_dim và cấu hình `max_hidden_layers`.

5. Vì sao không đưa Customer ID vào train?  
Vì Customer ID là mã định danh, không phản ánh hành vi khách hàng. Đưa ID vào train có thể làm sai lệch khoảng cách và kết quả cụm.

6. Vì sao cột chữ không dùng embedding?  
Categorical ngắn như Gender, Category, Payment Method là nhãn rời rạc nên dùng OneHotEncoder/OrdinalEncoder. Embedding chỉ phù hợp với text tự do dài hơn.

7. Vì sao dùng KMeans?  
KMeans đơn giản, phổ biến, phù hợp làm thuật toán phân cụm chính trên vector đặc trưng hoặc latent vector, dễ đánh giá bằng Silhouette và Inertia.

8. Vì sao phải chọn K tự động?  
Vì KMeans cần số cụm trước. Chọn K thủ công dễ chủ quan, nên hệ thống thử nhiều K và dùng metric để hỗ trợ lựa chọn.

9. t-SNE dùng để làm gì?  
t-SNE dùng để chiếu latent vector xuống 2D để quan sát. Nó không tạo cụm và không dùng để train.

10. Gom cụm dựa trên đặc trưng nào?  
Gom cụm dựa trên latent vector do Autoencoder học từ feature matrix sau preprocessing.

11. Làm sao diễn giải cụm nếu đã nén qua Autoencoder?  
Sau khi có nhãn cụm, hệ thống quay lại dữ liệu gốc để thống kê từng cụm bằng cluster profiling.

12. Autoencoder có cải thiện clustering không?  
Theo report test, có. Với `shopping_behavior_updated.csv`, Silhouette tăng từ khoảng 0,0974 lên 0,3742. Với `customer_shopping_data.csv`, tăng từ khoảng 0,1956 lên 0,4625.

13. Khách hàng mới đưa vào thì xử lý thế nào?  
Về nguyên tắc, khách hàng mới phải đi qua cùng preprocessing, Encoder tạo latent vector, sau đó gán vào tâm cụm KMeans gần nhất. Chức năng này chưa thấy có UI riêng trong project, cần kiểm tra lại trong project nếu muốn trình bày như chức năng đã hoàn thiện.

14. Hạn chế của hệ thống là gì?  
Kết quả phụ thuộc dataset và feature engineering, latent vector khó diễn giải trực tiếp, chưa có DEC hoàn chỉnh, chưa có explainability nâng cao và chưa triển khai production.

---

# VII. Ghi chú phần dùng chung và phần nhấn mạnh riêng

Phần dùng chung cho hai báo cáo gồm: mô tả đề tài, dataset, ứng dụng Streamlit, 7 tab, pipeline tổng thể, feature/metadata, preprocessing, KMeans, metric, output files và kết quả thực nghiệm.

Phần cần nhấn mạnh riêng cho Khai phá dữ liệu gồm: quy trình khai phá dữ liệu, EDA, data profiling, missing/duplicate/outlier/high cardinality, lựa chọn feature/metadata, KMeans là thuật toán phân cụm chính, tự động chọn K, đánh giá cụm và cluster profiling. Autoencoder trong báo cáo này là kỹ thuật hỗ trợ nâng cấp biểu diễn.

Phần cần nhấn mạnh riêng cho Deep Learning gồm: feature matrix sau preprocessing, Autoencoder, Encoder/Decoder, reconstruction loss, latent representation, KMeans trên latent vector, so sánh baseline KMeans trực tiếp với Autoencoder + KMeans và chứng minh Autoencoder giúp cải thiện clustering.

---

# VIII. Thông tin còn thiếu cần người dùng cung cấp thêm

1. Ảnh chụp giao diện 7 tab Streamlit để chèn vào Word.
2. Ảnh biểu đồ EDA cụ thể từ app: histogram, boxplot, heatmap, scatterplot, count plot, missing chart.
3. Ảnh loss curve và latent t-SNE từ lần chạy cuối cùng muốn dùng trong báo cáo.
4. Bảng cluster profiling cuối cùng nếu muốn diễn giải cụ thể từng cụm bằng số liệu mean/median/top category.
5. Quy định format Word của giảng viên: font, cỡ chữ, căn lề, số chương, số hình, số bảng.
6. Nếu muốn báo cáo ghi chức năng “gán cụm cho khách hàng mới” là đã có, cần kiểm tra lại trong project hoặc bổ sung chức năng trước.
