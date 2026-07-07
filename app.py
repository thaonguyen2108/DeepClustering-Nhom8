from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception as exc:
    raise RuntimeError("Ung dung nay can thu vien streamlit. Cai dat bang: pip install streamlit") from exc

from pipeline import (
    AutoencoderConfig,
    ClusteringConfig,
    EmbeddingConfig,
    EvaluationConfig,
    PreprocessingConfig,
    SchemaDetectionConfig,
    apply_column_role_overrides,
    apply_preprocessing_plan,
    cluster_latent_features,
    detect_schema,
    embed_text_features,
    ensure_original_index,
    estimate_cleaning_impact,
    evaluate_clustering,
    export_cluster_results,
    export_final_report,
    get_column_groups,
    load_tabular_dataset,
    preprocess_features,
    profile_data,
    profile_to_frame,
    recommend_preprocessing,
    sample_for_plot,
    train_autoencoder_features,
)


st.set_page_config(page_title="Hệ thống phân cụm khách hàng", layout="wide")


FILE_FORMAT_MAP = {
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".json": "json",
}

PIPELINE_STAGES: List[Tuple[str, str]] = [
    ("data_check", "Kiểm tra dữ liệu"),
    ("preprocessing", "Tiền xử lý dữ liệu"),
    ("embedding", "Sinh embedding / vector đặc trưng"),
    ("autoencoder", "Train Autoencoder"),
    ("latent", "Trích xuất vector tiềm ẩn"),
    ("k_selection", "Tự động chọn số cụm K"),
    ("clustering", "Phân cụm"),
    ("evaluation", "Đánh giá"),
    ("visualization", "Trực quan hóa"),
    ("export", "Xuất kết quả"),
]

CHART_HELP = {
    "histogram": (
        "**Biểu đồ phân phối (Histogram)**\n\n"
        "Mục đích: xem một cột số phân bố như thế nào và dữ liệu tập trung ở khoảng giá trị nào. "
        "Cách đọc: trục X là khoảng giá trị, trục Y là số bản ghi trong khoảng đó. "
        "Nên dùng cho Age, Income, Purchase Amount hoặc các đặc trưng số tương tự."
    ),
    "boxplot": (
        "**Biểu đồ hộp (Boxplot)**\n\n"
        "Mục đích: xem median, khoảng Q1-Q3 và outlier của dữ liệu số. "
        "Điểm nằm ngoài râu hộp thường là giá trị bất thường cần xem xét trước khi tiền xử lý."
    ),
    "correlation": (
        "**Bản đồ tương quan (Correlation Heatmap)**\n\n"
        "Mục đích: xem mức độ liên hệ tuyến tính giữa các cột số. "
        "Giá trị gần 1 là tương quan dương mạnh, gần -1 là tương quan âm mạnh, gần 0 là ít tương quan tuyến tính. "
        "Tương quan không đồng nghĩa với quan hệ nhân quả."
    ),
    "scatter": (
        "**Biểu đồ phân tán (Scatterplot)**\n\n"
        "Mục đích: xem quan hệ giữa hai biến số, phát hiện xu hướng, nhóm dữ liệu hoặc outlier. "
        "Mỗi điểm là một bản ghi/khách hàng."
    ),
    "count": (
        "**Biểu đồ tần suất nhóm (Count Plot / Bar Chart)**\n\n"
        "Mục đích: xem số lượng bản ghi theo từng nhóm/category. "
        "Nếu cột có quá nhiều giá trị khác nhau, hệ thống chỉ hiển thị top N."
    ),
    "missing": (
        "**Biểu đồ giá trị thiếu**\n\n"
        "Mục đích: xem cột nào có nhiều giá trị thiếu để quyết định impute, giữ lại hay loại bỏ."
    ),
    "k_selection": (
        "**Biểu đồ Silhouette theo số cụm K và Elbow/Inertia**\n\n"
        "Silhouette càng cao thường cho thấy cụm rõ hơn. Inertia thường giảm khi K tăng; điểm gãy giúp tham khảo số cụm phù hợp."
    ),
    "cluster_size": (
        "**Phân bố kích thước cụm**\n\n"
        "Biểu đồ cho biết mỗi cụm có bao nhiêu khách hàng. Nếu một cụm quá nhỏ hoặc quá lớn, cần xem lại đặc trưng hoặc số cụm K."
    ),
    "loss": (
        "**Loss tái tạo của Autoencoder**\n\n"
        "Đường loss giảm dần cho thấy Autoencoder đang học cách tái tạo vector đặc trưng tốt hơn qua từng epoch."
    ),
    "latent": (
        "**Không gian tiềm ẩn 2D**\n\n"
        "Biểu đồ t-SNE chiếu vector tiềm ẩn sau Autoencoder xuống 2 chiều, tô màu theo nhãn cụm để quan sát mức độ tách cụm."
    ),
}


class MemoryLogHandler(logging.Handler):
    def __init__(self, store: List[str]):
        super().__init__(level=logging.INFO)
        self.store = store

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.store.append(self.format(record))
        except Exception:
            return


def init_session_state() -> None:
    defaults = {
        "raw_df": None,
        "current_df": None,
        "original_file_name": None,
        "data_profile": None,
        "cleaning_plan": None,
        "cleaning_impact": None,
        "cleaning_result": None,
        "cleaning_confirmed": False,
        "pipeline_results": {},
        "pipeline_logs": [],
        "pipeline_status": _initial_pipeline_status(),
        "pipeline_error": None,
        "output_paths": {},
        "model_config": {},
        "feature_columns": [],
        "metadata_columns": [],
        "column_role_reasons": {},
        "column_role_overrides": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _initial_pipeline_status() -> Dict[str, Dict[str, str]]:
    return {
        key: {
            "stage": label,
            "status": "Chờ chạy",
            "started_at": "",
            "ended_at": "",
            "duration": "",
            "note": "",
            "_start_ts": "",
        }
        for key, label in PIPELINE_STAGES
    }


def reset_for_new_dataset(file_name: str, dataframe: pd.DataFrame) -> None:
    raw_df = ensure_original_index(dataframe)
    st.session_state["raw_df"] = raw_df
    st.session_state["current_df"] = raw_df.copy()
    st.session_state["original_file_name"] = file_name
    st.session_state["data_profile"] = profile_data(raw_df)
    st.session_state["feature_columns"] = list(st.session_state["data_profile"].feature_columns)
    st.session_state["metadata_columns"] = list(st.session_state["data_profile"].metadata_columns)
    st.session_state["column_role_reasons"] = dict(st.session_state["data_profile"].column_role_reasons)
    st.session_state["column_role_overrides"] = {}
    st.session_state["cleaning_plan"] = recommend_preprocessing(st.session_state["data_profile"])
    st.session_state["cleaning_impact"] = estimate_cleaning_impact(raw_df, st.session_state["cleaning_plan"])
    st.session_state["cleaning_result"] = None
    st.session_state["cleaning_confirmed"] = False
    st.session_state["pipeline_results"] = {}
    st.session_state["pipeline_logs"] = []
    st.session_state["pipeline_status"] = _initial_pipeline_status()
    st.session_state["pipeline_error"] = None
    st.session_state["output_paths"] = {}


def infer_file_format(file_name: str) -> Optional[str]:
    return FILE_FORMAT_MAP.get(Path(file_name).suffix.lower())


def load_source(uploaded_file: Any, file_path: str, source_mode: str) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str]]:
    try:
        if source_mode == "Tải tệp lên":
            if uploaded_file is None:
                return None, None, "Vui lòng chọn tệp dữ liệu."
            file_format = infer_file_format(uploaded_file.name)
            if file_format is None:
                return None, None, "Chỉ hỗ trợ CSV, XLSX hoặc JSON."
            result = load_tabular_dataset(uploaded_file.getvalue(), config=None if file_format is None else _ingestion_config(file_format))
            return uploaded_file.name, result.dataframe, None

        normalized_path = file_path.strip()
        if not normalized_path:
                return None, None, "Vui lòng nhập đường dẫn tệp."
        path_obj = Path(normalized_path).expanduser()
        if not path_obj.exists() or not path_obj.is_file():
            return None, None, "Đường dẫn không tồn tại hoặc không phải tệp hợp lệ."
        file_format = infer_file_format(path_obj.name)
        if file_format is None:
            return None, None, "Chỉ hỗ trợ CSV, XLSX hoặc JSON."
        result = load_tabular_dataset(str(path_obj.resolve()), config=_ingestion_config(file_format))
        return path_obj.name, result.dataframe, None
    except Exception as exc:
        return None, None, f"Không thể đọc dữ liệu: {exc}"


def _ingestion_config(file_format: str) -> Any:
    from pipeline import IngestionConfig

    return IngestionConfig(file_format=file_format)


def render_load_tab() -> None:
    st.subheader("Tải dữ liệu")
    st.caption(
        "Tab này dùng để tải dữ liệu khách hàng vào hệ thống. Sau khi tải, hệ thống hiển thị thông tin tổng quan "
        "và bản xem trước dữ liệu."
    )
    source_mode = st.radio("Nguồn dữ liệu", ["Tải tệp lên", "Nhập đường dẫn"], horizontal=True)
    uploaded_file = None
    file_path = ""
    if source_mode == "Tải tệp lên":
        uploaded_file = st.file_uploader("Chọn tệp CSV, XLSX hoặc JSON", type=["csv", "xlsx", "json"])
    else:
        file_path = st.text_input("Nhập đường dẫn tệp dữ liệu")

    if st.button("Đọc và xem trước dữ liệu", type="primary"):
        file_name, dataframe, error_message = load_source(uploaded_file, file_path, source_mode)
        if error_message:
            st.error(error_message)
        elif dataframe is None or dataframe.empty:
            st.error("Dữ liệu rỗng hoặc không đọc được.")
        else:
            reset_for_new_dataset(file_name or "dataset", dataframe)
            st.success(f"Đã đọc dữ liệu: {file_name}")

    raw_df = st.session_state.get("raw_df")
    profile = st.session_state.get("data_profile")
    if raw_df is None or profile is None:
        st.info("Hãy tải dataset hoặc nhập đường dẫn để bắt đầu.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nguồn dữ liệu", st.session_state.get("original_file_name") or "")
    col2.metric("Số dòng", profile.row_count)
    col3.metric("Số cột", profile.column_count)
    col4.metric("Missing cells", profile.total_missing_cells)

    type_col1, type_col2, type_col3 = st.columns(3)
    type_col1.metric("Cột numeric", len(profile.numeric_columns))
    type_col2.metric("Cột categorical", len(profile.categorical_columns))
    type_col3.metric("Cột text", len(profile.text_columns))

    st.markdown("**Bản xem trước dữ liệu**")
    st.dataframe(raw_df.head(20), width="stretch")

    st.markdown("**Danh sách cột và kiểu dữ liệu**")
    st.dataframe(profile_to_frame(profile)[["name", "dtype", "inferred_type", "missing_count", "unique_count"]], width="stretch")


def render_eda_tab() -> None:
    df = st.session_state.get("raw_df")
    profile = st.session_state.get("data_profile")
    if df is None or profile is None:
        st.info("Vui lòng tải dữ liệu trước.")
        return

    st.subheader("Tổng quan & trực quan hóa dữ liệu")
    st.caption(
        "Tab này giúp khám phá dữ liệu ban đầu thông qua thống kê và biểu đồ. Mục tiêu là hiểu dữ liệu trước "
        "khi tiền xử lý và huấn luyện mô hình."
    )
    summary_cols = st.columns(4)
    summary_cols[0].metric("Số dòng", profile.row_count)
    summary_cols[1].metric("Số cột", profile.column_count)
    summary_cols[2].metric("Missing cells", profile.total_missing_cells)
    summary_cols[3].metric("Duplicate rows", profile.duplicate_row_count)

    with st.expander("Profile theo cot", expanded=True):
        st.dataframe(profile_to_frame(profile), width="stretch")

    feature_columns = set(st.session_state.get("feature_columns") or profile.feature_columns)
    numeric_columns = [column for column in profile.numeric_columns if column in feature_columns]
    categorical_columns = [column for column in profile.categorical_columns if column in feature_columns]
    plot_df, sampled = sample_for_plot(df, max_rows=5000)
    if sampled:
        st.info("Biểu đồ đang dùng sample 5.000 dòng để tránh làm app bị treo.")

    chart_tabs = st.tabs(["Histogram", "Boxplot", "Correlation", "Scatter", "Count plot", "Missing chart"])
    with chart_tabs[0]:
        st.markdown(CHART_HELP["histogram"])
        if not numeric_columns:
            st.warning("Không có cột numeric để vẽ histogram.")
        else:
            column = st.selectbox("Cột numeric", numeric_columns, key="hist_col")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(pd.to_numeric(plot_df[column], errors="coerce").dropna(), bins=30, color="#4c78a8", alpha=0.85)
            ax.set_title(f"Biểu đồ phân phối - {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Tan suat")
            st.pyplot(fig)
            plt.close(fig)

    with chart_tabs[1]:
        st.markdown(CHART_HELP["boxplot"])
        if not numeric_columns:
            st.warning("Không có cột numeric để vẽ boxplot.")
        else:
            column = st.selectbox("Cột numeric", numeric_columns, key="box_col")
            group = st.selectbox("Nhóm categorical", ["Không nhóm"] + categorical_columns, key="box_group")
            fig, ax = plt.subplots(figsize=(9, 4))
            if group == "Không nhóm":
                ax.boxplot(pd.to_numeric(plot_df[column], errors="coerce").dropna(), vert=False)
                ax.set_yticklabels([column])
            else:
                grouped_values = [
                    pd.to_numeric(values[column], errors="coerce").dropna()
                    for _, values in plot_df.groupby(group)
                ][:20]
                labels = [str(label)[:20] for label in plot_df[group].dropna().astype(str).unique()[:20]]
                ax.boxplot(grouped_values, labels=labels, vert=True)
                ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"Biểu đồ hộp - {column}")
            st.pyplot(fig)
            plt.close(fig)

    with chart_tabs[2]:
        st.markdown(CHART_HELP["correlation"])
        if len(numeric_columns) < 2:
            st.warning("Cần ít nhất 2 cột numeric để vẽ correlation heatmap.")
        else:
            method = st.selectbox("Correlation method", ["pearson", "spearman"], key="corr_method")
            corr = plot_df[numeric_columns].apply(pd.to_numeric, errors="coerce").corr(method=method)
            fig, ax = plt.subplots(figsize=(8, 6))
            image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index)
            fig.colorbar(image, ax=ax)
            ax.set_title(f"Bản đồ tương quan ({method})")
            st.pyplot(fig)
            plt.close(fig)

    with chart_tabs[3]:
        st.markdown(CHART_HELP["scatter"])
        if len(numeric_columns) < 2:
            st.warning("Cần ít nhất 2 cột numeric để vẽ scatterplot.")
        else:
            x_col = st.selectbox("Cột X", numeric_columns, key="scatter_x")
            y_options = [col for col in numeric_columns if col != x_col] or numeric_columns
            y_col = st.selectbox("Cột Y", y_options, key="scatter_y")
            color_col = st.selectbox("Màu theo", ["Không"] + categorical_columns, key="scatter_color")
            fig, ax = plt.subplots(figsize=(7, 5))
            if color_col == "Không":
                ax.scatter(plot_df[x_col], plot_df[y_col], s=16, alpha=0.65)
            else:
                for label, group_df in plot_df.groupby(color_col):
                    ax.scatter(group_df[x_col], group_df[y_col], s=16, alpha=0.65, label=str(label)[:20])
                ax.legend(fontsize=8)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Biểu đồ phân tán")
            st.pyplot(fig)
            plt.close(fig)

    with chart_tabs[4]:
        st.markdown(CHART_HELP["count"])
        if not categorical_columns:
            st.warning("Không có cột categorical để vẽ count plot.")
        else:
            column = st.selectbox("Cột categorical", categorical_columns, key="bar_col")
            top_n = st.slider("Top N", min_value=5, max_value=50, value=20)
            counts = plot_df[column].astype(str).value_counts().head(top_n)
            if profile_to_frame(profile).set_index("name").loc[column, "unique_count"] > top_n:
                st.warning(f"Cột có cardinality cao, chỉ hiển thị top {top_n}.")
            fig, ax = plt.subplots(figsize=(9, 4))
            counts.sort_values().plot(kind="barh", ax=ax, color="#59a14f")
            ax.set_title(f"Biểu đồ tần suất nhóm - {column}")
            st.pyplot(fig)
            plt.close(fig)

    with chart_tabs[5]:
        st.markdown(CHART_HELP["missing"])
        missing_frame = profile_to_frame(profile)[["name", "missing_count", "missing_ratio"]]
        missing_frame = missing_frame[missing_frame["missing_count"] > 0]
        if missing_frame.empty:
            st.success("Dữ liệu không có missing value.")
        else:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(missing_frame["name"], missing_frame["missing_count"], color="#e15759")
            ax.tick_params(axis="x", rotation=45)
            ax.set_title("Giá trị thiếu theo cột")
            ax.set_ylabel("Số missing")
            st.pyplot(fig)
            plt.close(fig)

    with st.expander("Thống kê numeric"):
        st.dataframe(pd.DataFrame(profile.numeric_summary).T, width="stretch")
    with st.expander("Thống kê categorical"):
        st.dataframe(pd.DataFrame(profile.categorical_summary).T, width="stretch")


def render_preprocessing_tab() -> None:
    df = st.session_state.get("raw_df")
    profile = st.session_state.get("data_profile")
    if df is None or profile is None:
        st.info("Vui lòng tải dữ liệu trước.")
        return

    st.subheader("Kiểm tra & tiền xử lý dữ liệu")
    st.caption(
        "Tab này kiểm tra chất lượng dữ liệu, phát hiện missing value, duplicate, outlier và đề xuất cách xử lý. "
        "Người dùng cần xác nhận trước khi pipeline chính được chạy."
    )
    plan = dict(st.session_state.get("cleaning_plan") or recommend_preprocessing(profile))

    st.markdown("**Phân loại cột đặc trưng và metadata**")
    st.info(
        "Các cột metadata/identifier như Customer ID, ID hoặc cột chuỗi có unique_ratio rất cao sẽ không được đưa "
        "vào feature matrix để train Autoencoder/KMeans, nhưng vẫn được giữ trong kết quả để đối chiếu."
    )
    role_overrides: Dict[str, str] = {}
    role_rows = []
    for column_profile in profile.column_profiles:
        default_role = "metadata" if column_profile.name in st.session_state.get("metadata_columns", []) else "feature"
        selected_role = st.selectbox(
            f"Vai trò cột: {column_profile.name}",
            ["feature", "metadata"],
            index=0 if default_role == "feature" else 1,
            key=f"role_{column_profile.name}",
        )
        role_overrides[column_profile.name] = selected_role
        role_rows.append(
            {
                "Cột": column_profile.name,
                "Kiểu suy luận": column_profile.inferred_type,
                "Unique ratio": round(float(column_profile.unique_ratio), 4),
                "Vai trò": selected_role,
                "Lý do": profile.column_role_reasons.get(column_profile.name, column_profile.role_reason),
            }
        )

    feature_columns, metadata_columns, role_reasons = apply_column_role_overrides(df, profile, role_overrides)
    st.session_state["column_role_overrides"] = role_overrides
    st.session_state["feature_columns"] = feature_columns
    st.session_state["metadata_columns"] = metadata_columns
    st.session_state["column_role_reasons"] = role_reasons
    st.dataframe(pd.DataFrame(role_rows), width="stretch", hide_index=True)
    if profile.suspicious_high_uniqueness_columns:
        st.warning(
            "Các cột chuỗi có tỷ lệ unique cao cần xem xét như metadata: "
            + ", ".join(profile.suspicious_high_uniqueness_columns)
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        plan["missing_numeric"] = st.selectbox(
            "Missing numeric",
            ["median", "mean", "drop_rows", "keep"],
            index=["median", "mean", "drop_rows", "keep"].index(plan.get("missing_numeric", "median")),
        )
        plan["duplicates"] = st.selectbox(
            "Duplicate rows",
            ["drop", "keep"],
            index=["drop", "keep"].index(plan.get("duplicates", "drop")),
        )
        plan["encoding"] = st.selectbox(
            "Encoding categorical",
            ["onehot", "ordinal", "drop", "keep"],
            index=["onehot", "ordinal", "drop", "keep"].index(plan.get("encoding", "onehot")),
            disabled=not profile.categorical_columns,
        )
    with col2:
        plan["missing_categorical"] = st.selectbox(
            "Missing categorical",
            ["mode", "unknown", "drop_rows", "keep"],
            index=["mode", "unknown", "drop_rows", "keep"].index(plan.get("missing_categorical", "mode")),
        )
        plan["outliers"] = st.selectbox(
            "Outlier numeric",
            ["keep", "cap_iqr", "drop_iqr", "drop_zscore"],
            index=["keep", "cap_iqr", "drop_iqr", "drop_zscore"].index(plan.get("outliers", "keep")),
            disabled=not profile.numeric_columns,
        )
        plan["scaling"] = st.selectbox(
            "Scaling numeric",
            ["standard", "minmax", "robust", "none"],
            index=["standard", "minmax", "robust", "none"].index(plan.get("scaling", "standard")),
        )
    with col3:
        plan["missing_text"] = st.selectbox(
            "Missing text",
            ["empty", "unknown", "drop_rows", "keep"],
            index=["empty", "unknown", "drop_rows", "keep"].index(plan.get("missing_text", "empty")),
            disabled=not profile.text_columns,
        )
        plan["text"] = st.selectbox(
            "Text embedding",
            ["use", "ignore", "drop"],
            index=["use", "ignore", "drop"].index(plan.get("text", "ignore")),
            disabled=not profile.text_columns,
        )
        plan["drop_constant_columns"] = st.checkbox(
            "Drop cột hằng / gần hằng",
            value=bool(plan.get("drop_constant_columns", True)),
        )

    if profile.categorical_columns and plan["encoding"] == "onehot":
        high_cardinality = [
            column.name
            for column in profile.column_profiles
            if column.inferred_type == "categorical" and column.unique_count > 100
        ]
        if high_cardinality:
            st.warning("Một số cột categorical có cardinality cao. Nên dùng OrdinalEncoder: " + ", ".join(high_cardinality))
    if plan["scaling"] == "none":
        st.warning("Không scale dữ liệu không được khuyến nghị cho Autoencoder/KMeans.")

    impact = estimate_cleaning_impact(df, plan)
    st.session_state["cleaning_plan"] = plan
    st.session_state["cleaning_impact"] = impact

    st.markdown("**Tóm tắt trước khi xác nhận**")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Dòng ban đầu", impact["initial_rows"])
    metric_cols[1].metric("Dự kiến giữ", impact["estimated_rows_kept"])
    metric_cols[2].metric("Dự kiến drop", impact["estimated_rows_dropped"])
    metric_cols[3].metric("Cột drop", impact["estimated_columns_dropped"])
    metric_cols[4].metric("Missing impute", impact["missing_values_to_impute"])
    metric_cols[5].metric("Outlier cap", impact["outlier_values_to_cap"])

    st.dataframe(profile_to_frame(profile), width="stretch")

    if st.button("Xác nhận xử lý và tiếp tục", type="primary"):
        try:
            cleaning_result = apply_preprocessing_plan(df, plan, export=True)
            st.session_state["cleaning_result"] = cleaning_result
            st.session_state["current_df"] = cleaning_result.cleaned_dataframe
            st.session_state["cleaning_confirmed"] = True
            st.session_state["output_paths"].update(
                {
                    "removed_rows": cleaning_result.removed_rows_path,
                    "cleaning_report": cleaning_result.report_path,
                }
            )
            st.success("Đã xác nhận và áp dụng kế hoạch tiền xử lý.")
        except Exception as exc:
            st.error(f"Không thể áp dụng kế hoạch tiền xử lý: {exc}")

    result = st.session_state.get("cleaning_result")
    if result is not None:
        st.markdown("**Dữ liệu sau tiền xử lý**")
        st.dataframe(result.cleaned_dataframe.head(20), width="stretch")
        if result.removed_rows_path:
            st.info(f"Dòng dữ liệu bị loại đã lưu tại: {result.removed_rows_path}")
        if result.report_path:
            st.info(f"Báo cáo tiền xử lý đã lưu tại: {result.report_path}")


def render_model_config_tab() -> None:
    current_df = st.session_state.get("current_df")
    n_samples = 0 if current_df is None else int(len(current_df))
    k_config = prepare_k_widget_state(n_samples)
    feature_columns = [column for column in st.session_state.get("feature_columns", []) if current_df is not None and column in current_df.columns]
    input_dim_estimate = max(1, len(feature_columns))
    default_latent_dim = min(8, max(2, input_dim_estimate // 4))
    if input_dim_estimate > 2:
        default_latent_dim = min(default_latent_dim, input_dim_estimate - 1)
    st.subheader("Cấu hình mô hình")
    st.caption(
        "Tab này dùng để cấu hình Autoencoder, số chiều latent, số epoch, batch size và cách chọn số cụm K."
    )
    st.info(
        f"Ước lượng đặc trưng đầu vào trước Autoencoder: {input_dim_estimate} cột feature. "
        f"Metadata/identifier không được dùng để train: {', '.join(st.session_state.get('metadata_columns', [])) or 'không có'}."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        latent_mode = st.selectbox("Chế độ latent_dim", ["Tự động đề xuất", "Tự nhập"], index=0)
        latent_dim = st.number_input(
            "latent_dim thủ công",
            min_value=2,
            max_value=512,
            value=int(default_latent_dim),
            disabled=latent_mode == "Tự động đề xuất",
        )
        epochs = st.number_input("epochs", min_value=1, max_value=500, value=50)
        batch_size = st.number_input("batch_size", min_value=1, max_value=1024, value=64)
        learning_rate = st.number_input("learning_rate", min_value=0.00001, max_value=1.0, value=0.001, format="%.5f")
        effective_latent_dim = default_latent_dim if latent_mode == "Tự động đề xuất" else int(latent_dim)
        st.metric("Tỷ lệ nén dự kiến", f"{input_dim_estimate} → {effective_latent_dim}")
        if effective_latent_dim >= input_dim_estimate:
            st.warning(
                "Latent dimension đang lớn hơn hoặc bằng input dimension. Khi đó Autoencoder không thể hiện rõ vai trò nén đặc trưng."
            )
    with col2:
        auto_k = st.selectbox("Chọn số cụm K", ["Tự động chọn K", "Tự nhập K"])
        if not k_config["is_clusterable"]:
            st.warning(
                "Dữ liệu hiện tại có quá ít dòng để phân cụm. Cần ít nhất 3 dòng để chạy KMeans với K >= 2."
            )
            k_min = 2
            k_max = 2
            manual_k = 2
        else:
            k_min = st.number_input(
                "k_min",
                min_value=2,
                max_value=int(k_config["max_allowed_k"]),
                key="model_k_min",
                disabled=auto_k != "Tự động chọn K",
            )
            if st.session_state["model_k_max"] < int(k_min):
                st.session_state["model_k_max"] = int(k_min)
            k_max = st.number_input(
                "k_max",
                min_value=int(k_min),
                max_value=int(k_config["max_allowed_k"]),
                key="model_k_max",
                disabled=auto_k != "Tự động chọn K",
            )
            manual_k = st.number_input(
                "K thủ công",
                min_value=2,
                max_value=int(k_config["max_allowed_k"]),
                key="model_manual_k",
                disabled=auto_k == "Tự động chọn K",
            )
        random_state = st.number_input("random_state", min_value=0, max_value=999999, value=42)
    with col3:
        compute_tsne = st.checkbox("Tính t-SNE", value=True)
        tsne_perplexity = st.slider("t-SNE perplexity", min_value=2, max_value=50, value=30)
        visualization_sample = st.number_input("Sample tối đa cho trực quan hóa", min_value=100, max_value=50000, value=5000)

    if k_config["is_clusterable"] and int(k_max) < int(k_min):
        st.warning("k_max phải lớn hơn hoặc bằng k_min.")

    st.session_state["model_config"] = {
        "latent_dim": "auto" if latent_mode == "Tự động đề xuất" else int(latent_dim),
        "display_latent_dim": int(effective_latent_dim),
        "input_dim_estimate": int(input_dim_estimate),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "auto_k": auto_k == "Tự động chọn K",
        "k_min": int(k_min),
        "k_max": int(k_max),
        "manual_k": int(manual_k),
        "random_state": int(random_state),
        "compute_tsne": bool(compute_tsne),
        "tsne_perplexity": float(tsne_perplexity),
        "visualization_sample": int(visualization_sample),
    }


def prepare_k_widget_state(n_samples: int) -> Dict[str, Any]:
    if n_samples < 3:
        for key in ("model_k_min", "model_k_max", "model_manual_k"):
            st.session_state[key] = 2
        return {
            "is_clusterable": False,
            "max_allowed_k": 2,
            "k_min": 2,
            "k_max": 2,
            "manual_k": 2,
        }

    max_allowed_k = min(10, int(n_samples) - 1)
    default_k_max = min(max_allowed_k, max(2, int(np.sqrt(max(2, n_samples)))))

    k_min = _clamp_int(st.session_state.get("model_k_min", 2), 2, max_allowed_k)
    k_max = _clamp_int(st.session_state.get("model_k_max", default_k_max), 2, max_allowed_k)
    if k_max < k_min:
        k_max = k_min
    manual_k = _clamp_int(st.session_state.get("model_manual_k", 2), 2, max_allowed_k)

    st.session_state["model_k_min"] = k_min
    st.session_state["model_k_max"] = k_max
    st.session_state["model_manual_k"] = manual_k
    return {
        "is_clusterable": True,
        "max_allowed_k": max_allowed_k,
        "k_min": k_min,
        "k_max": k_max,
        "manual_k": manual_k,
    }


def _clamp_int(value: Any, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = lower
    return max(lower, min(parsed, upper))


def render_progress_tab() -> None:
    st.subheader("Tiến trình xử lý")
    st.caption(
        "Tab này hiển thị từng bước chạy của pipeline Deep Clustering, từ tiền xử lý đến Autoencoder, phân cụm, "
        "đánh giá và xuất kết quả."
    )
    current_df = st.session_state.get("current_df")
    if current_df is None:
        st.info("Vui lòng tải dữ liệu trước.")
        return
    if not st.session_state.get("cleaning_confirmed"):
        st.warning("Cần xác nhận tiền xử lý trước khi chạy pipeline chính.")

    if st.button("Chạy pipeline Deep Clustering", type="primary", disabled=not st.session_state.get("cleaning_confirmed")):
        run_pipeline()

    render_status_table()
    if st.session_state.get("pipeline_error"):
        st.error(st.session_state["pipeline_error"])
    if st.session_state.get("pipeline_logs"):
        with st.expander("Pipeline logs"):
            st.code("\n".join(st.session_state["pipeline_logs"]), language="text")


def run_pipeline() -> None:
    st.session_state["pipeline_results"] = {}
    st.session_state["pipeline_logs"] = []
    st.session_state["pipeline_status"] = _initial_pipeline_status()
    st.session_state["pipeline_error"] = None
    logger = create_pipeline_logger(st.session_state["pipeline_logs"])
    current_df = st.session_state["current_df"]
    model_config = st.session_state.get("model_config") or {}
    cleaning_plan = st.session_state.get("cleaning_plan") or {}
    feature_columns = [column for column in st.session_state.get("feature_columns", []) if column in current_df.columns]
    metadata_columns = [column for column in st.session_state.get("metadata_columns", []) if column in current_df.columns]
    if not feature_columns:
        st.session_state["pipeline_error"] = "Không có cột đặc trưng hợp lệ để huấn luyện Autoencoder/KMeans."
        return
    feature_df = current_df[feature_columns].copy()

    try:
        _set_stage("data_check", "Đang chạy")
        schema = detect_schema(feature_df, config=SchemaDetectionConfig(), upstream_metadata={"source": "feature_dataframe"})
        _set_stage(
            "data_check",
            "Hoàn thành",
            f"Dữ liệu sau tiền xử lý: {len(current_df)} dòng. Feature columns: {len(feature_columns)}. Metadata columns: {len(metadata_columns)}.",
        )

        _set_stage("preprocessing", "Đang chạy")
        preprocessing_result = preprocess_features(
            dataframe=feature_df,
            schema=schema,
            config=PreprocessingConfig(
                numeric_scaler=_map_scaler(cleaning_plan.get("scaling", "standard")),
                categorical_encoder=_map_encoder(cleaning_plan.get("encoding", "onehot")),
                include_text_columns=cleaning_plan.get("text", "use") == "use",
            ),
            logger=logger,
        )
        feature_shape = tuple(preprocessing_result.X_numeric.shape)
        _set_stage("preprocessing", "Hoàn thành", f"Feature matrix trước Autoencoder: {feature_shape}")

        _set_stage("embedding", "Đang chạy")
        embedding_result = embed_text_features(
            X_text=preprocessing_result.X_text,
            feature_map=preprocessing_result.feature_map,
            preprocessing_metadata=preprocessing_result.feature_metadata,
            config=EmbeddingConfig(
                batch_size=32,
                model_name="intfloat/multilingual-e5-small",
            ),
            logger=logger,
        )
        _set_stage("embedding", "Hoàn thành", f"Embedding shape: {tuple(embedding_result.X_embedding.shape)}")

        _set_stage("autoencoder", "Đang chạy")
        autoencoder_result = train_autoencoder_features(
            X_numeric=preprocessing_result.X_numeric,
            X_embedding=embedding_result.X_embedding,
            config=AutoencoderConfig(
                latent_dim=model_config.get("latent_dim", "auto"),
                epochs=int(model_config.get("epochs", 50)),
                batch_size=int(model_config.get("batch_size", 64)),
                learning_rate=float(model_config.get("learning_rate", 0.001)),
                random_seed=int(model_config.get("random_state", 42)),
            ),
            logger=logger,
        )
        input_dim = int(autoencoder_result.autoencoder_report.input_shape[1])
        latent_dim = int(autoencoder_result.autoencoder_report.latent_dim)
        final_loss = autoencoder_result.training_metadata.get("final_loss")
        _set_stage(
            "autoencoder",
            "Hoàn thành",
            f"Feature matrix: {autoencoder_result.autoencoder_report.input_shape}. Tỷ lệ nén: {input_dim} → {latent_dim}. Final loss: {final_loss}.",
        )
        _set_stage("latent", "Hoàn thành", f"Latent vector sau Encoder: {autoencoder_result.Z_latent.shape}")

        _set_stage("k_selection", "Đang chạy")
        if model_config.get("auto_k", True):
            k_min = int(model_config.get("k_min", 2))
            k_max = int(model_config.get("k_max", 10))
        else:
            k_min = k_max = int(model_config.get("manual_k", 2))
        k_max = min(k_max, max(2, len(feature_df) - 1))
        k_min = min(k_min, k_max)
        clustering_result = cluster_latent_features(
            Z_latent=autoencoder_result.Z_latent,
            config=ClusteringConfig(
                k_min=k_min,
                k_max=k_max,
                random_state=int(model_config.get("random_state", 42)),
            ),
            logger=logger,
        )
        _set_stage("k_selection", "Hoàn thành", f"Số cụm K được chọn: {clustering_result.best_k}")
        _set_stage("clustering", "Hoàn thành", f"Đã phân cụm {len(clustering_result.cluster_labels)} dòng.")

        _set_stage("evaluation", "Đang chạy")
        evaluation_result = evaluate_clustering(
            Z_latent=autoencoder_result.Z_latent,
            cluster_labels=clustering_result.cluster_labels,
            config=EvaluationConfig(
                compute_tsne=bool(model_config.get("compute_tsne", True)),
                perplexity=float(model_config.get("tsne_perplexity", 30.0)),
                random_state=int(model_config.get("random_state", 42)),
            ),
            logger=logger,
        )
        metrics_note = (
            f"Silhouette={evaluation_result.evaluation_metrics['silhouette_score']:.4f}, "
            f"Davies-Bouldin={evaluation_result.evaluation_metrics['davies_bouldin']}, "
            f"Calinski-Harabasz={evaluation_result.evaluation_metrics['calinski_harabasz']}"
        )
        _set_stage("evaluation", "Hoàn thành", metrics_note)
        _set_stage("visualization", "Hoàn thành", "Đã chuẩn bị dữ liệu trực quan hóa latent space.")

        _set_stage("export", "Đang chạy")
        result_frame = current_df.copy()
        result_frame["cluster_label"] = clustering_result.cluster_labels
        result_path = export_cluster_results(result_frame)
        final_report_path = export_final_report(
            {
                "dataset": {
                    "source": st.session_state.get("original_file_name"),
                    "rows": int(len(current_df)),
                    "columns": int(current_df.shape[1]),
                },
                "feature_columns": feature_columns,
                "metadata_columns": metadata_columns,
                "suspicious_high_uniqueness_columns": []
                if st.session_state.get("data_profile") is None
                else st.session_state["data_profile"].suspicious_high_uniqueness_columns,
                "column_role_reasons": st.session_state.get("column_role_reasons", {}),
                "cleaning_report": None
                if st.session_state.get("cleaning_result") is None
                else st.session_state["cleaning_result"].report,
                "model_config": model_config,
                "autoencoder": {
                    "input_dim": input_dim,
                    "latent_dim": latent_dim,
                    "compression_ratio": f"{input_dim} -> {latent_dim}",
                    "loss_history": autoencoder_result.training_metadata.get("loss_history", []),
                },
                "selected_k": int(clustering_result.best_k),
                "metrics": evaluation_result.evaluation_metrics,
                "result_path": result_path,
            }
        )
        _set_stage("export", "Hoàn thành", f"Kết quả phân cụm: {result_path}. Final report: {final_report_path}.")

        st.session_state["pipeline_results"] = {
            "schema": schema,
            "preprocessing": preprocessing_result,
            "embedding": embedding_result,
            "autoencoder": autoencoder_result,
            "clustering": clustering_result,
            "evaluation": evaluation_result,
            "result_frame": result_frame,
        }
        st.session_state["output_paths"].update({"cluster_results": result_path, "final_report": final_report_path})
        st.success("Đã hoàn thành pipeline Deep Clustering.")
    except Exception as exc:
        st.session_state["pipeline_error"] = f"Pipeline dừng vì lỗi: {exc}"
        running = [key for key, value in st.session_state["pipeline_status"].items() if value["status"] == "Đang chạy"]
        if running:
            _set_stage(running[-1], "Lỗi", str(exc))


def create_pipeline_logger(log_store: List[str]) -> logging.Logger:
    logger = logging.getLogger(f"streamlit_pipeline.{time.time_ns()}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = MemoryLogHandler(log_store)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def _set_stage(stage_key: str, status: str, note: str = "") -> None:
    status_map = st.session_state["pipeline_status"]
    now_time = time.strftime("%H:%M:%S")
    now_ts = time.perf_counter()
    status_map[stage_key]["status"] = status
    status_map[stage_key]["note"] = note
    if status == "Đang chạy":
        status_map[stage_key]["started_at"] = now_time
        status_map[stage_key]["_start_ts"] = str(now_ts)
        status_map[stage_key]["duration"] = "0.00s"
    if status in {"Hoàn thành", "Lỗi", "Bỏ qua"}:
        status_map[stage_key]["ended_at"] = now_time
        try:
            duration = max(0.0, now_ts - float(status_map[stage_key].get("_start_ts") or now_ts))
        except Exception:
            duration = 0.0
        status_map[stage_key]["duration"] = f"{duration:.2f}s"
    st.session_state["pipeline_status"] = status_map


def render_status_table() -> None:
    status_frame = pd.DataFrame(st.session_state["pipeline_status"].values())
    display_frame = status_frame.drop(columns=["_start_ts"], errors="ignore")
    display_frame = display_frame.rename(
        columns={
            "stage": "Tên stage",
            "status": "Trạng thái",
            "started_at": "Bắt đầu",
            "ended_at": "Kết thúc",
            "duration": "Thời lượng",
            "note": "Ghi chú",
        }
    )
    done_count = int((status_frame["status"].isin(["Hoàn thành", "Lỗi", "Bỏ qua"])).sum())
    st.progress(done_count / len(PIPELINE_STAGES))
    st.dataframe(display_frame, width="stretch", hide_index=True)


def render_evaluation_tab() -> None:
    results = st.session_state.get("pipeline_results") or {}
    if not results:
        st.info("Vui lòng chạy pipeline trước.")
        return
    st.subheader("Đánh giá kết quả")
    st.caption(
        "Tab này hiển thị các chỉ số và biểu đồ đánh giá chất lượng phân cụm, đồng thời trực quan hóa latent space "
        "sau Autoencoder."
    )
    evaluation_result = results["evaluation"]
    clustering_result = results["clustering"]
    autoencoder_result = results["autoencoder"]
    metrics = evaluation_result.evaluation_metrics
    removed_count = 0
    if st.session_state.get("cleaning_result") is not None:
        removed_count = int(len(st.session_state["cleaning_result"].removed_rows))

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Selected K", clustering_result.best_k)
    col2.metric("Silhouette", round(float(metrics["silhouette_score"]), 6))
    col3.metric("Davies-Bouldin", "-" if metrics["davies_bouldin"] is None else round(float(metrics["davies_bouldin"]), 6))
    col4.metric("Calinski-Harabasz", "-" if metrics["calinski_harabasz"] is None else round(float(metrics["calinski_harabasz"]), 6))
    col5.metric("Số cụm", len(evaluation_result.cluster_distribution))
    col6.metric("Dòng bị loại", removed_count)

    eval_tabs = st.tabs(["Chọn K", "Kích thước cụm", "Autoencoder", "Latent 2D"])
    with eval_tabs[0]:
        st.markdown(CHART_HELP["k_selection"])
        metadata = clustering_result.clustering_metadata
        score_frame = pd.DataFrame(
            {
                "K": list(metadata.get("silhouette_scores", {}).keys()),
                "Silhouette": list(metadata.get("silhouette_scores", {}).values()),
                "Inertia": [metadata.get("inertia_values", {}).get(k) for k in metadata.get("silhouette_scores", {}).keys()],
            }
        ).sort_values("K")
        st.dataframe(score_frame, width="stretch")
        if not score_frame.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(score_frame["K"], score_frame["Silhouette"], marker="o")
            axes[0].axvline(clustering_result.best_k, color="red", linestyle="--")
            axes[0].set_title("Biểu đồ Silhouette theo số cụm K")
            axes[1].plot(score_frame["K"], score_frame["Inertia"], marker="o", color="#f28e2b")
            axes[1].axvline(clustering_result.best_k, color="red", linestyle="--")
            axes[1].set_title("Biểu đồ Elbow/Inertia")
            st.pyplot(fig)
            plt.close(fig)
    with eval_tabs[1]:
        st.markdown(CHART_HELP["cluster_size"])
        distribution = pd.DataFrame(
            {"cluster_label": list(evaluation_result.cluster_distribution.keys()), "count": list(evaluation_result.cluster_distribution.values())}
        ).sort_values("cluster_label")
        st.dataframe(distribution, width="stretch")
        st.bar_chart(distribution.set_index("cluster_label"))
    with eval_tabs[2]:
        st.markdown("### Đánh giá Autoencoder")
        st.markdown(CHART_HELP["loss"])
        loss_history = autoencoder_result.training_metadata.get("loss_history", [])
        if not loss_history:
            st.info("Chưa có lịch sử loss của Autoencoder. Vui lòng chạy pipeline trước.")
        else:
            st.line_chart(pd.DataFrame({"reconstruction_loss": loss_history}))
        st.info("Histogram lỗi tái tạo theo từng dòng là nâng cấp tiếp theo nếu cần tính reconstruction_error chi tiết.")
    with eval_tabs[3]:
        st.markdown(CHART_HELP["latent"])
        z_2d = evaluation_result.visualization_data.get("Z_2d")
        labels = np.asarray(evaluation_result.visualization_data.get("labels"))
        if z_2d is None:
            st.info("t-SNE đang tắt hoặc không đủ dữ liệu.")
        else:
            render_latent_scatter(np.asarray(z_2d), labels)


def render_latent_scatter(z_2d: np.ndarray, labels: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    unique_labels = np.unique(labels)
    color_map = plt.cm.get_cmap("tab10", max(2, len(unique_labels)))
    for index, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], s=26, alpha=0.75, label=f"Cluster {label}", color=color_map(index))
    ax.set_title("Trực quan hóa không gian tiềm ẩn 2D")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_results_tab() -> None:
    results = st.session_state.get("pipeline_results") or {}
    if not results:
        st.info("Vui lòng chạy pipeline trước.")
        return
    result_frame = results["result_frame"]
    st.subheader("Kết quả phân cụm")
    st.caption(
        "Tab này hiển thị nhãn cụm của từng khách hàng, thống kê đặc điểm từng cụm và cho phép tải kết quả phân cụm."
    )
    st.dataframe(result_frame, width="stretch")

    distribution = result_frame["cluster_label"].value_counts().sort_index().rename_axis("cluster_label").reset_index(name="count")
    st.markdown("**Số lượng mỗi cụm**")
    st.dataframe(distribution, width="stretch")
    st.bar_chart(distribution.set_index("cluster_label"))

    st.markdown("**Cluster profiling**")
    feature_columns = [column for column in st.session_state.get("feature_columns", []) if column in result_frame.columns]
    feature_frame = result_frame[feature_columns + ["cluster_label"]].copy()
    groups = get_column_groups(feature_frame.drop(columns=["cluster_label"], errors="ignore"))
    if groups["numeric"]:
        numeric_profile = result_frame.groupby("cluster_label")[groups["numeric"]].agg(["mean", "median", "min", "max"])
        st.dataframe(numeric_profile, width="stretch")
    if groups["categorical"]:
        cat_summary = []
        for cluster_label, group_df in result_frame.groupby("cluster_label"):
            row = {"cluster_label": cluster_label}
            for column in groups["categorical"][:10]:
                mode_values = group_df[column].dropna().mode()
                row[f"{column}_top"] = "" if mode_values.empty else mode_values.iloc[0]
            cat_summary.append(row)
        st.dataframe(pd.DataFrame(cat_summary), width="stretch")

    insight = build_simple_cluster_insight(result_frame, groups["numeric"])
    if insight:
        st.info(insight)

    csv_data = result_frame.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Tải kết quả CSV", data=csv_data, file_name="customer_segments.csv", mime="text/csv")
    for label, path in st.session_state.get("output_paths", {}).items():
        if path:
            st.caption(f"{label}: {path}")


def build_simple_cluster_insight(result_frame: pd.DataFrame, numeric_columns: List[str]) -> str:
    if not numeric_columns:
        return "Chưa có đủ đặc trưng số phù hợp để tạo nhận xét cụm. Hãy xem bảng thống kê cụm để diễn giải chính xác hơn."
    means = result_frame.groupby("cluster_label")[numeric_columns].mean(numeric_only=True)
    notes = []
    for column in numeric_columns[:5]:
        if column in means and means[column].notna().any():
            cluster_id = means[column].idxmax()
            notes.append(f"Cụm {cluster_id} có giá trị trung bình cao hơn ở đặc trưng `{column}`.")
    if not notes:
        return "Chưa đủ cơ sở để tạo nhận xét cụm tự động. Cần xem thêm bảng thống kê cụm để diễn giải chính xác."
    return " ".join(notes) + " Đây là nhận xét thống kê đơn giản, không nên xem như kết luận kinh doanh tuyệt đối."


def _map_scaler(value: str) -> str:
    return {"standard": "standard", "minmax": "minmax", "robust": "robust", "none": "standard"}.get(value, "standard")


def _map_encoder(value: str) -> str:
    return {"onehot": "onehot", "ordinal": "label", "keep": "onehot", "drop": "onehot"}.get(value, "onehot")


def main() -> None:
    init_session_state()
    st.title("Hệ thống phân cụm khách hàng tự động")
    st.caption("Deep Clustering dựa trên Autoencoder, kết hợp EDA và tiền xử lý cho Khai phá dữ liệu.")

    tabs = st.tabs(
        [
            "1. Tải dữ liệu",
            "2. Tổng quan & trực quan hóa dữ liệu",
            "3. Kiểm tra & tiền xử lý dữ liệu",
            "4. Cấu hình mô hình",
            "5. Tiến trình xử lý",
            "6. Đánh giá kết quả",
            "7. Kết quả phân cụm",
        ]
    )
    with tabs[0]:
        render_load_tab()
    with tabs[1]:
        render_eda_tab()
    with tabs[2]:
        render_preprocessing_tab()
    with tabs[3]:
        render_model_config_tab()
    with tabs[4]:
        render_progress_tab()
    with tabs[5]:
        render_evaluation_tab()
    with tabs[6]:
        render_results_tab()


if __name__ == "__main__":
    main()
