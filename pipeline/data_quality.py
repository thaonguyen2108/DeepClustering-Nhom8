from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


OUTPUT_ROOT = Path("outputs")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: str
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    is_constant: bool = False
    is_near_constant: bool = False
    outlier_count: int = 0
    outlier_ratio: float = 0.0
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    top_value: Optional[str] = None
    top_frequency: Optional[int] = None
    role: str = "feature"
    role_reason: str = "Có thể dùng làm đặc trưng phân cụm."


@dataclass
class DataProfile:
    row_count: int
    column_count: int
    total_missing_cells: int
    duplicate_row_count: int
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    metadata_columns: List[str] = field(default_factory=list)
    suspicious_high_uniqueness_columns: List[str] = field(default_factory=list)
    column_role_reasons: Dict[str, str] = field(default_factory=dict)
    column_profiles: List[ColumnProfile] = field(default_factory=list)
    numeric_summary: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    categorical_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class CleaningExportResult:
    cleaned_dataframe: pd.DataFrame
    removed_rows: pd.DataFrame
    report: Dict[str, Any]
    removed_rows_path: Optional[str] = None
    report_path: Optional[str] = None


def ensure_original_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    if "original_index" not in frame.columns:
        frame.insert(0, "original_index", frame.index)
    return frame


def infer_simple_type(series: pd.Series) -> str:
    if is_numeric_dtype(series):
        return "numeric"
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    as_text = non_null.astype(str)
    unique_ratio = float(as_text.nunique(dropna=True) / max(1, len(non_null)))
    avg_length = float(as_text.str.len().mean())
    avg_tokens = float(as_text.str.split().str.len().mean())
    if avg_length >= 25 or avg_tokens >= 4 or unique_ratio >= 0.30:
        return "text"
    return "categorical"


def get_column_groups(dataframe: pd.DataFrame) -> Dict[str, List[str]]:
    groups = {"numeric": [], "categorical": [], "text": [], "unknown": []}
    for column in dataframe.columns:
        if column == "original_index":
            continue
        inferred = infer_simple_type(dataframe[column])
        groups.setdefault(inferred, []).append(str(column))
    return groups


def classify_column_role(column_name: str, series: pd.Series, inferred_type: str) -> Tuple[str, str, bool]:
    normalized_name = column_name.strip().lower().replace(" ", "_").replace("-", "_")
    non_null_count = int(series.notna().sum())
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = float(unique_count / max(1, non_null_count))
    id_tokens = ("id", "customerid", "customer_id", "user_id", "client_id", "invoice", "order_id")
    date_tokens = ("date", "time", "datetime", "timestamp")

    if column_name == "original_index":
        return "metadata", "Chỉ dùng để truy vết dòng dữ liệu gốc.", False
    if normalized_name in id_tokens or normalized_name.endswith("_id"):
        return "metadata", "Tên cột giống mã định danh, không nên dùng để huấn luyện Autoencoder/KMeans.", False
    if any(token in normalized_name for token in date_tokens):
        return "metadata", "Cột ngày/giờ dạng thô nên được giữ làm metadata hoặc trích xuất đặc trưng thời gian trước khi huấn luyện.", False
    if inferred_type in {"categorical", "text"} and unique_ratio >= 0.90 and unique_count >= max(10, int(non_null_count * 0.5)):
        return "metadata", "Cột dạng chuỗi có tỷ lệ giá trị duy nhất rất cao, có khả năng là mã định danh hoặc metadata.", True
    return "feature", "Có thể dùng làm đặc trưng phân cụm.", False


def apply_column_role_overrides(
    dataframe: pd.DataFrame,
    profile: DataProfile,
    role_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    role_overrides = role_overrides or {}
    feature_columns: List[str] = []
    metadata_columns: List[str] = []
    reasons: Dict[str, str] = dict(profile.column_role_reasons)
    available_columns = [str(column) for column in dataframe.columns if str(column) != "original_index"]

    for column in available_columns:
        role = role_overrides.get(column)
        if role is None:
            role = "metadata" if column in profile.metadata_columns else "feature"
        if role == "metadata":
            metadata_columns.append(column)
            if column in role_overrides:
                reasons[column] = "Người dùng chọn cột này là metadata/identifier."
        else:
            feature_columns.append(column)
            if column in role_overrides:
                reasons[column] = "Người dùng chọn cột này là đặc trưng đầu vào."

    return feature_columns, metadata_columns, reasons


def detect_outlier_masks(dataframe: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> Dict[str, pd.Series]:
    columns = numeric_columns or [
        str(column)
        for column in dataframe.columns
        if column != "original_index" and is_numeric_dtype(dataframe[column])
    ]
    masks: Dict[str, pd.Series] = {}
    for column in columns:
        numeric_series = pd.to_numeric(dataframe[column], errors="coerce")
        non_null = numeric_series.dropna()
        if len(non_null) < 4:
            masks[column] = pd.Series(False, index=dataframe.index)
            continue
        q1 = float(non_null.quantile(0.25))
        q3 = float(non_null.quantile(0.75))
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            masks[column] = pd.Series(False, index=dataframe.index)
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        masks[column] = (numeric_series < lower) | (numeric_series > upper)
    return masks


def profile_data(dataframe: pd.DataFrame) -> DataProfile:
    frame = ensure_original_index(dataframe)
    groups = get_column_groups(frame)
    outlier_masks = detect_outlier_masks(frame, groups["numeric"])
    column_profiles: List[ColumnProfile] = []

    for column in frame.columns:
        if column == "original_index":
            continue
        series = frame[column]
        inferred_type = infer_simple_type(series)
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = float(unique_count / max(1, int(series.notna().sum())))
        non_null = series.dropna()
        top_value = None
        top_frequency = None
        if not non_null.empty:
            value_counts = non_null.astype(str).value_counts()
            if not value_counts.empty:
                top_value = str(value_counts.index[0])
                top_frequency = int(value_counts.iloc[0])

        outlier_count = 0
        outlier_ratio = 0.0
        lower_bound = None
        upper_bound = None
        if inferred_type == "numeric":
            mask = outlier_masks.get(str(column), pd.Series(False, index=frame.index))
            outlier_count = int(mask.sum())
            outlier_ratio = float(outlier_count / max(1, len(frame)))
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric_series) >= 4:
                q1 = float(numeric_series.quantile(0.25))
                q3 = float(numeric_series.quantile(0.75))
                iqr = q3 - q1
                if np.isfinite(iqr) and iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

        is_constant = unique_count <= 1
        is_near_constant = bool(top_frequency is not None and top_frequency / max(1, len(frame)) >= 0.98)
        role, role_reason, suspicious_high_uniqueness = classify_column_role(str(column), series, inferred_type)
        column_profiles.append(
            ColumnProfile(
                name=str(column),
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_ratio=float(missing_count / max(1, len(frame))),
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                is_constant=is_constant,
                is_near_constant=is_near_constant,
                outlier_count=outlier_count,
                outlier_ratio=outlier_ratio,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                top_value=top_value,
                top_frequency=top_frequency,
                role=role,
                role_reason=role_reason,
            )
        )

    numeric_summary: Dict[str, Dict[str, Optional[float]]] = {}
    for column in groups["numeric"]:
        numeric_series = pd.to_numeric(frame[column], errors="coerce")
        numeric_summary[column] = {
            "mean": _safe_float(numeric_series.mean()),
            "median": _safe_float(numeric_series.median()),
            "std": _safe_float(numeric_series.std()),
            "min": _safe_float(numeric_series.min()),
            "q1": _safe_float(numeric_series.quantile(0.25)),
            "q3": _safe_float(numeric_series.quantile(0.75)),
            "max": _safe_float(numeric_series.max()),
        }

    categorical_summary: Dict[str, Dict[str, Any]] = {}
    for column in groups["categorical"]:
        series = frame[column].dropna().astype(str)
        value_counts = series.value_counts()
        categorical_summary[column] = {
            "unique_count": int(series.nunique()),
            "top_value": None if value_counts.empty else str(value_counts.index[0]),
            "top_frequency": 0 if value_counts.empty else int(value_counts.iloc[0]),
        }

    feature_columns = [column.name for column in column_profiles if column.role == "feature"]
    metadata_columns = [column.name for column in column_profiles if column.role == "metadata"]
    suspicious_high_uniqueness_columns = [
        column.name
        for column in column_profiles
        if "tỷ lệ giá trị duy nhất rất cao" in column.role_reason
    ]

    return DataProfile(
        row_count=int(len(frame)),
        column_count=int(frame.shape[1]),
        total_missing_cells=int(frame.drop(columns=["original_index"], errors="ignore").isna().sum().sum()),
        duplicate_row_count=int(frame.drop(columns=["original_index"], errors="ignore").duplicated().sum()),
        numeric_columns=groups["numeric"],
        categorical_columns=groups["categorical"],
        text_columns=groups["text"],
        feature_columns=feature_columns,
        metadata_columns=metadata_columns,
        suspicious_high_uniqueness_columns=suspicious_high_uniqueness_columns,
        column_role_reasons={column.name: column.role_reason for column in column_profiles},
        column_profiles=column_profiles,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
    )


def recommend_preprocessing(profile: DataProfile) -> Dict[str, Any]:
    has_outliers = any(column.outlier_count > 0 for column in profile.column_profiles)
    missing_numeric = "median" if has_outliers else "mean"
    if profile.row_count > 0:
        missing_ratio_total = profile.total_missing_cells / max(1, profile.row_count * max(1, profile.column_count))
        if 0 < missing_ratio_total <= 0.01:
            missing_numeric = "drop_rows"

    max_cardinality = 0
    for column in profile.column_profiles:
        if column.inferred_type == "categorical":
            max_cardinality = max(max_cardinality, column.unique_count)

    return {
        "missing_numeric": missing_numeric,
        "missing_categorical": "mode" if profile.total_missing_cells <= max(1, profile.row_count * 0.01) else "unknown",
        "missing_text": "empty",
        "duplicates": "drop",
        "outliers": "cap_iqr" if has_outliers else "keep",
        "encoding": "ordinal" if max_cardinality > 100 else "onehot",
        "scaling": "robust" if has_outliers else "standard",
        "text": "use" if profile.text_columns else "ignore",
        "drop_constant_columns": True,
    }


def estimate_cleaning_impact(dataframe: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    frame = ensure_original_index(dataframe)
    row_drop_mask = pd.Series(False, index=frame.index)
    duplicate_mask = frame.drop(columns=["original_index"], errors="ignore").duplicated()
    if plan.get("duplicates") == "drop":
        row_drop_mask |= duplicate_mask

    missing_plan = {
        "numeric": plan.get("missing_numeric"),
        "categorical": plan.get("missing_categorical"),
        "text": plan.get("missing_text"),
    }
    groups = get_column_groups(frame)
    for group_name, strategy in missing_plan.items():
        if strategy == "drop_rows":
            columns = groups.get(group_name, [])
            if columns:
                row_drop_mask |= frame[columns].isna().any(axis=1)

    outlier_masks = detect_outlier_masks(frame, groups["numeric"])
    if plan.get("outliers") == "drop_iqr":
        for mask in outlier_masks.values():
            row_drop_mask |= mask
    elif plan.get("outliers") == "drop_zscore":
        for column in groups["numeric"]:
            numeric_series = pd.to_numeric(frame[column], errors="coerce")
            std = float(numeric_series.std() or 0.0)
            if std > 0:
                row_drop_mask |= ((numeric_series - numeric_series.mean()).abs() / std) > 3

    drop_columns = _resolve_drop_columns(frame, plan)
    return {
        "initial_rows": int(len(frame)),
        "estimated_rows_kept": int((~row_drop_mask).sum()),
        "estimated_rows_dropped": int(row_drop_mask.sum()),
        "estimated_columns_dropped": int(len(drop_columns)),
        "drop_columns": drop_columns,
        "missing_values_to_impute": int(_count_imputed_values(frame, plan)),
        "outlier_values_to_cap": int(_count_outliers(frame) if plan.get("outliers") == "cap_iqr" else 0),
        "duplicate_rows_to_drop": int(duplicate_mask.sum() if plan.get("duplicates") == "drop" else 0),
    }


def apply_preprocessing_plan(
    dataframe: pd.DataFrame,
    plan: Dict[str, Any],
    *,
    export: bool = True,
) -> CleaningExportResult:
    frame = ensure_original_index(dataframe)
    original_frame = frame.copy()
    removed_reasons: Dict[Any, List[Dict[str, str]]] = {}
    actions: List[Dict[str, Any]] = []

    def mark_removed(mask: pd.Series, reason: str, action: str, affected_column: Optional[str] = None) -> None:
        for index_value in frame.index[mask.fillna(False)]:
            removed_reasons.setdefault(index_value, []).append(
                {
                    "reason": reason,
                    "action": action,
                    "affected_column": affected_column or "",
                }
            )

    duplicate_mask = frame.drop(columns=["original_index"], errors="ignore").duplicated()
    if plan.get("duplicates") == "drop" and duplicate_mask.any():
        mark_removed(duplicate_mask, "duplicate_row", "drop_row")
        actions.append({"type": "drop_duplicates", "rows": int(duplicate_mask.sum())})

    groups = get_column_groups(frame)
    for group_name, strategy_key in [
        ("numeric", "missing_numeric"),
        ("categorical", "missing_categorical"),
        ("text", "missing_text"),
    ]:
        strategy = plan.get(strategy_key, "keep")
        for column in groups.get(group_name, []):
            missing_mask = frame[column].isna()
            if not missing_mask.any() or strategy == "keep":
                continue
            if strategy == "drop_rows":
                mark_removed(missing_mask, "missing_value", "drop_row", column)
                actions.append({"type": "drop_missing_rows", "column": column, "rows": int(missing_mask.sum())})
            else:
                fill_value = _resolve_fill_value(frame[column], group_name, strategy)
                frame.loc[missing_mask, column] = fill_value
                actions.append(
                    {
                        "type": "impute_missing",
                        "column": column,
                        "strategy": strategy,
                        "count": int(missing_mask.sum()),
                        "fill_value": None if pd.isna(fill_value) else str(fill_value),
                    }
                )

    outlier_strategy = plan.get("outliers", "keep")
    outlier_masks = detect_outlier_masks(frame, groups["numeric"])
    if outlier_strategy == "cap_iqr":
        for column, mask in outlier_masks.items():
            if not mask.any():
                continue
            numeric_series = pd.to_numeric(frame[column], errors="coerce")
            q1 = float(numeric_series.quantile(0.25))
            q3 = float(numeric_series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            frame[column] = numeric_series.clip(lower=lower, upper=upper)
            actions.append(
                {
                    "type": "cap_outliers_iqr",
                    "column": column,
                    "count": int(mask.sum()),
                    "lower": lower,
                    "upper": upper,
                }
            )
    elif outlier_strategy == "drop_iqr":
        for column, mask in outlier_masks.items():
            if mask.any():
                mark_removed(mask, "outlier_iqr", "drop_row", column)
                actions.append({"type": "drop_outlier_rows_iqr", "column": column, "rows": int(mask.sum())})
    elif outlier_strategy == "drop_zscore":
        for column in groups["numeric"]:
            numeric_series = pd.to_numeric(frame[column], errors="coerce")
            std = float(numeric_series.std() or 0.0)
            if std <= 0:
                continue
            mask = ((numeric_series - numeric_series.mean()).abs() / std) > 3
            if mask.any():
                mark_removed(mask, "outlier_zscore", "drop_row", column)
                actions.append({"type": "drop_outlier_rows_zscore", "column": column, "rows": int(mask.sum())})

    drop_columns = _resolve_drop_columns(frame, plan)
    if drop_columns:
        frame = frame.drop(columns=drop_columns, errors="ignore")
        actions.append({"type": "drop_columns", "columns": drop_columns})

    removed_indices = sorted(removed_reasons.keys())
    removed_rows = _build_removed_rows(original_frame, removed_indices, removed_reasons)
    if removed_indices:
        frame = frame.drop(index=removed_indices, errors="ignore").reset_index(drop=True)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "initial_rows": int(len(original_frame)),
        "final_rows": int(len(frame)),
        "removed_row_count": int(len(removed_rows)),
        "dropped_columns": drop_columns,
        "actions": actions,
        "plan": plan,
    }

    removed_path = None
    report_path = None
    if export:
        removed_path = export_removed_rows(removed_rows)
        report_path = export_cleaning_report(report)

    return CleaningExportResult(
        cleaned_dataframe=frame,
        removed_rows=removed_rows,
        report=report,
        removed_rows_path=removed_path,
        report_path=report_path,
    )


def export_removed_rows(removed_rows: pd.DataFrame) -> Optional[str]:
    if removed_rows.empty:
        return None
    output_dir = OUTPUT_ROOT / "removed_rows"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"removed_rows_{_timestamp()}.csv"
    removed_rows.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def export_cleaning_report(report: Dict[str, Any]) -> str:
    output_dir = OUTPUT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"data_cleaning_report_{_timestamp()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def export_cluster_results(result_frame: pd.DataFrame) -> str:
    output_dir = OUTPUT_ROOT / "clustering_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"customer_segments_{_timestamp()}.csv"
    result_frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def export_final_report(report: Dict[str, Any]) -> str:
    output_dir = OUTPUT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"final_report_{_timestamp()}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def profile_to_frame(profile: DataProfile) -> pd.DataFrame:
    return pd.DataFrame([asdict(column) for column in profile.column_profiles])


def sample_for_plot(dataframe: pd.DataFrame, max_rows: int = 5000, random_state: int = 42) -> Tuple[pd.DataFrame, bool]:
    if len(dataframe) <= max_rows:
        return dataframe, False
    return dataframe.sample(n=max_rows, random_state=random_state), True


def _resolve_drop_columns(frame: pd.DataFrame, plan: Dict[str, Any]) -> List[str]:
    drop_columns: List[str] = []
    if plan.get("drop_constant_columns", False):
        for column in frame.columns:
            if column == "original_index":
                continue
            if frame[column].nunique(dropna=True) <= 1:
                drop_columns.append(str(column))
    if plan.get("encoding") == "drop":
        drop_columns.extend(get_column_groups(frame).get("categorical", []))
    if plan.get("text") == "drop":
        drop_columns.extend(get_column_groups(frame).get("text", []))
    return sorted(set(drop_columns))


def _count_imputed_values(frame: pd.DataFrame, plan: Dict[str, Any]) -> int:
    total = 0
    groups = get_column_groups(frame)
    for group_name, strategy_key in [
        ("numeric", "missing_numeric"),
        ("categorical", "missing_categorical"),
        ("text", "missing_text"),
    ]:
        strategy = plan.get(strategy_key, "keep")
        if strategy not in {"keep", "drop_rows"}:
            total += int(frame[groups.get(group_name, [])].isna().sum().sum()) if groups.get(group_name) else 0
    return total


def _count_outliers(frame: pd.DataFrame) -> int:
    return int(sum(mask.sum() for mask in detect_outlier_masks(frame).values()))


def _resolve_fill_value(series: pd.Series, group_name: str, strategy: str) -> Any:
    if group_name == "numeric":
        numeric_series = pd.to_numeric(series, errors="coerce")
        if strategy == "median":
            return float(numeric_series.median())
        if strategy == "mean":
            return float(numeric_series.mean())
        return 0.0
    if strategy == "mode":
        mode_values = series.dropna().mode()
        return "unknown" if mode_values.empty else mode_values.iloc[0]
    if strategy == "empty":
        return ""
    return "unknown"


def _build_removed_rows(
    original_frame: pd.DataFrame,
    removed_indices: List[Any],
    removed_reasons: Dict[Any, List[Dict[str, str]]],
) -> pd.DataFrame:
    if not removed_indices:
        return pd.DataFrame()
    rows = original_frame.loc[removed_indices].copy()
    reasons = []
    actions = []
    affected_columns = []
    for index_value in removed_indices:
        events = removed_reasons.get(index_value, [])
        reasons.append("; ".join(sorted({event["reason"] for event in events})))
        actions.append("; ".join(sorted({event["action"] for event in events})))
        affected_columns.append("; ".join(sorted({event["affected_column"] for event in events if event["affected_column"]})))
    rows.insert(1, "reason", reasons)
    rows.insert(2, "action", actions)
    rows.insert(3, "affected_column", affected_columns)
    return rows.reset_index(drop=True)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value) or not np.isfinite(float(value)):
            return None
        return float(value)
    except Exception:
        return None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
