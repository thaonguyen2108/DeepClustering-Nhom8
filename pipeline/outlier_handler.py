from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .schema_detection import DatasetSchema, SchemaColumn


ConfirmationCallback = Callable[["OutlierHandlingReport"], bool]


@dataclass
class OutlierHandlerConfig:
    outlier_strategy: str = "cap"
    fallback_strategy: str = "ignore"
    column_strategy_overrides: Dict[str, str] = field(default_factory=dict)
    iqr_multiplier: float = 1.5
    min_non_null_values: int = 4
    preview_only: bool = False
    require_confirmation: bool = True
    auto_confirm: bool = True
    confirmation_callback: Optional[ConfirmationCallback] = None
    include_columns_without_outliers: bool = True
    include_row_preview: bool = True
    sample_row_count: int = 5
    row_preview_column_limit: Optional[int] = None
    reset_index_after_drop: bool = True
    outlier_flag_column: str = "outlier_flag"
    group_flag_mode: str = "boolean"
    outlier_group_value: str = "OUTLIER"
    overwrite_existing_flag_column: bool = False


@dataclass
class OutlierColumnReport:
    name: str
    inferred_type: str
    non_null_count: int
    outlier_count: int
    outlier_ratio: float
    outlier_ratio_non_null: float
    configured_strategy: str
    resolved_strategy: str
    requires_action: bool
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)
    applied_strategy: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    action_status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlierHandlingReport:
    stage: str
    rows_before: int
    columns_before: int
    numeric_columns_in_schema: int
    numeric_columns_analyzed: int
    total_outlier_values: int
    total_affected_rows: int
    affected_columns: List[str]
    column_reports: List[OutlierColumnReport] = field(default_factory=list)
    strategy_summary: Dict[str, List[str]] = field(default_factory=dict)
    confirmation_required: bool = True
    confirmation_status: str = "pending"
    preview_generated: bool = True
    applied: bool = False
    rows_after: Optional[int] = None
    columns_after: Optional[int] = None
    dropped_rows: int = 0
    flag_column_added: Optional[str] = None
    fallback_events: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class OutlierHandlerResult:
    dataframe: pd.DataFrame
    report: OutlierHandlingReport


class OutlierHandlerStageError(RuntimeError):
    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        try:
            super().__init__(message)
            self.cause = cause
        except Exception as exc:
            raise RuntimeError("Failed to initialize OutlierHandlerStageError.") from exc


class OutlierHandlerModule:
    stage_name = "OUTLIER_HANDLER"
    _supported_types = {"numeric", "categorical", "text", "unknown"}
    _strategy_aliases = {
        "cap": "cap",
        "winsorize": "cap",
        "winsorization": "cap",
        "drop": "drop",
        "drop_rows": "drop",
        "drop rows": "drop",
        "remove": "drop",
        "group": "group",
        "outlier_group": "group",
        "ignore": "ignore",
        "none": "ignore",
        "skip": "ignore",
    }

    def __init__(
        self,
        config: Optional[OutlierHandlerConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        try:
            self.config = config or OutlierHandlerConfig()
            self.logger = logger or logging.getLogger(__name__)
        except Exception as exc:
            raise OutlierHandlerStageError("Failed to initialize outlier handler module.", cause=exc) from exc

    def generate_report(
        self,
        dataframe: pd.DataFrame,
        schema: Any,
        config: Optional[OutlierHandlerConfig] = None,
    ) -> OutlierHandlingReport:
        try:
            active_config = config or self.config
            validated_frame, schema_lookup, warnings = self._validate_inputs(dataframe, schema)
            self._log(
                logging.INFO,
                f"Stage start | rows={validated_frame.shape[0]} | columns={validated_frame.shape[1]}",
            )

            numeric_schema_columns = [
                schema_column
                for schema_column in schema_lookup.values()
                if schema_column.inferred_type == "numeric" and schema_column.name in validated_frame.columns
            ]

            column_reports: List[OutlierColumnReport] = []
            fallback_events: List[Dict[str, Any]] = []
            total_outlier_values = 0
            combined_outlier_mask = pd.Series(False, index=validated_frame.index)

            for schema_column in numeric_schema_columns:
                column_report, outlier_mask = self._build_column_report(
                    dataframe=validated_frame,
                    column_name=schema_column.name,
                    schema_column=schema_column,
                    config=active_config,
                    warnings=warnings,
                    fallback_events=fallback_events,
                )
                if active_config.include_columns_without_outliers or column_report.outlier_count > 0:
                    column_reports.append(column_report)

                total_outlier_values += int(column_report.outlier_count)
                combined_outlier_mask = combined_outlier_mask | outlier_mask

            affected_columns = [report.name for report in column_reports if report.outlier_count > 0]
            strategy_summary = self._build_strategy_summary(column_reports)
            total_affected_rows = int(combined_outlier_mask.sum())

            self._log(
                logging.INFO,
                f"Outlier summary | numeric_columns={len(numeric_schema_columns)} | "
                f"affected_columns={len(affected_columns)} | affected_rows={total_affected_rows}",
            )

            if fallback_events:
                self._log(logging.WARNING, f"Fallback usage | count={len(fallback_events)}")

            report = OutlierHandlingReport(
                stage=self.stage_name,
                rows_before=int(validated_frame.shape[0]),
                columns_before=int(validated_frame.shape[1]),
                numeric_columns_in_schema=int(len(numeric_schema_columns)),
                numeric_columns_analyzed=int(len(numeric_schema_columns)),
                total_outlier_values=int(total_outlier_values),
                total_affected_rows=total_affected_rows,
                affected_columns=affected_columns,
                column_reports=column_reports,
                strategy_summary=strategy_summary,
                confirmation_required=bool(active_config.require_confirmation),
                confirmation_status="pending",
                preview_generated=True,
                applied=False,
                fallback_events=fallback_events,
                warnings=warnings,
            )

            self._log(logging.INFO, f"Strategy preview | {self._format_strategy_summary(strategy_summary)}")
            return report
        except OutlierHandlerStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | report generation failed with {type(exc).__name__}: {exc}")
            raise OutlierHandlerStageError("Outlier report generation failed.", cause=exc) from exc

    def run(
        self,
        dataframe: pd.DataFrame,
        schema: Any,
        config: Optional[OutlierHandlerConfig] = None,
    ) -> OutlierHandlerResult:
        try:
            active_config = config or self.config
            report = self.generate_report(dataframe=dataframe, schema=schema, config=active_config)
            confirmed = self._resolve_confirmation(report, active_config)

            if active_config.preview_only or not confirmed:
                preview_frame = dataframe.copy(deep=True)
                report.applied = False
                report.rows_after = int(preview_frame.shape[0])
                report.columns_after = int(preview_frame.shape[1])
                for column_report in report.column_reports:
                    if column_report.outlier_count > 0:
                        column_report.action_status = (
                            "preview_only" if active_config.preview_only else "awaiting_confirmation"
                        )
                    else:
                        column_report.action_status = "no_outliers"
                return OutlierHandlerResult(dataframe=preview_frame, report=report)

            cleaned_frame, updated_report = self._apply_strategies(
                dataframe=dataframe,
                report=report,
                config=active_config,
            )
            return OutlierHandlerResult(dataframe=cleaned_frame, report=updated_report)
        except OutlierHandlerStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage failed with {type(exc).__name__}: {exc}")
            raise OutlierHandlerStageError("Outlier handling failed.", cause=exc) from exc

    def _validate_inputs(
        self,
        dataframe: pd.DataFrame,
        schema: Any,
    ) -> Tuple[pd.DataFrame, Dict[str, SchemaColumn], List[str]]:
        try:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(f"Expected pandas DataFrame, received {type(dataframe).__name__}.")
            if dataframe.shape[1] == 0:
                raise ValueError("Input DataFrame does not contain any columns.")

            warnings: List[str] = []
            schema_columns = self._extract_schema_columns(schema, warnings)
            schema_lookup = {column.name: column for column in schema_columns}

            dataframe_columns = {str(column) for column in dataframe.columns}
            for schema_name in list(schema_lookup):
                if schema_name not in dataframe_columns:
                    warnings.append(f"Schema column '{schema_name}' is not present in the DataFrame.")

            return dataframe, schema_lookup, warnings
        except Exception as exc:
            self._log(logging.ERROR, f"Error | input validation failed: {exc}")
            raise OutlierHandlerStageError("Invalid input for outlier handler.", cause=exc) from exc

    def _extract_schema_columns(self, schema: Any, warnings: List[str]) -> List[SchemaColumn]:
        try:
            if isinstance(schema, DatasetSchema):
                return list(schema.columns)

            if isinstance(schema, dict):
                raw_columns = schema.get("columns", [])
            else:
                raw_columns = getattr(schema, "columns", None)

            if raw_columns is None:
                raise TypeError("Schema input must provide a 'columns' collection.")

            schema_columns: List[SchemaColumn] = []
            for raw_column in raw_columns:
                if isinstance(raw_column, SchemaColumn):
                    schema_columns.append(raw_column)
                    continue

                if isinstance(raw_column, dict):
                    schema_columns.append(
                        SchemaColumn(
                            name=str(raw_column.get("name")),
                            inferred_type=str(raw_column.get("inferred_type", "unknown")),
                            metadata=dict(raw_column.get("metadata", {})),
                        )
                    )
                    continue

                name = getattr(raw_column, "name", None)
                inferred_type = getattr(raw_column, "inferred_type", "unknown")
                metadata = getattr(raw_column, "metadata", {})
                if name is None:
                    warnings.append("Encountered schema column without a name; skipping entry.")
                    continue
                schema_columns.append(
                    SchemaColumn(name=str(name), inferred_type=str(inferred_type), metadata=dict(metadata or {}))
                )

            if not schema_columns:
                raise ValueError("Schema does not contain any usable column definitions.")

            return schema_columns
        except Exception as exc:
            self._log(logging.ERROR, f"Error | schema extraction failed: {exc}")
            raise OutlierHandlerStageError("Failed to extract schema columns.", cause=exc) from exc

    def _build_column_report(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        schema_column: SchemaColumn,
        config: OutlierHandlerConfig,
        warnings: List[str],
        fallback_events: List[Dict[str, Any]],
    ) -> Tuple[OutlierColumnReport, pd.Series]:
        try:
            series = dataframe[column_name]
            raw_strategy = config.column_strategy_overrides.get(column_name, config.outlier_strategy)
            try:
                configured_strategy = self._resolve_configured_strategy(column_name, config)
                resolved_strategy = configured_strategy
                fallback_used = False
                fallback_reason: Optional[str] = None
            except Exception as exc:
                resolved_strategy = self._resolve_fallback_strategy(column_name, config, warnings)
                configured_strategy = str(raw_strategy)
                fallback_used = True
                fallback_reason = str(exc)
                fallback_events.append(
                    {
                        "column": column_name,
                        "original_strategy": configured_strategy,
                        "fallback_strategy": resolved_strategy,
                        "reason": fallback_reason,
                    }
                )
                self._log(
                    logging.WARNING,
                    f"Fallback usage | column={column_name} | from={configured_strategy} | "
                    f"to={resolved_strategy} | reason={fallback_reason}",
                )

            converted_series = pd.to_numeric(series, errors="coerce")
            non_null_mask = converted_series.notna()
            non_null_count = int(non_null_mask.sum())

            base_report = OutlierColumnReport(
                name=column_name,
                inferred_type="numeric",
                non_null_count=non_null_count,
                outlier_count=0,
                outlier_ratio=0.0,
                outlier_ratio_non_null=0.0,
                configured_strategy=str(configured_strategy),
                resolved_strategy=resolved_strategy,
                requires_action=False,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                action_status="pending",
                metadata={
                    "original_dtype": str(series.dtype),
                    "schema_confidence": schema_column.metadata.get("inference_confidence"),
                    "converted_non_null_count": non_null_count,
                },
            )

            if non_null_count < config.min_non_null_values:
                warning_message = (
                    f"Column '{column_name}' skipped outlier detection because it has fewer than "
                    f"{config.min_non_null_values} numeric values."
                )
                warnings.append(warning_message)
                self._log(logging.WARNING, f"Fallback usage | column={column_name} | reason=insufficient_numeric_values")
                base_report.resolved_strategy = "ignore"
                base_report.fallback_used = True
                base_report.fallback_reason = "insufficient_numeric_values"
                base_report.action_status = "skipped"
                return base_report, pd.Series(False, index=dataframe.index)

            q1 = float(converted_series[non_null_mask].quantile(0.25))
            q3 = float(converted_series[non_null_mask].quantile(0.75))
            iqr = float(q3 - q1)
            lower_bound = float(q1 - config.iqr_multiplier * iqr)
            upper_bound = float(q3 + config.iqr_multiplier * iqr)

            if any(pd.isna(value) for value in [q1, q3, iqr, lower_bound, upper_bound]):
                warning_message = f"Column '{column_name}' produced invalid IQR bounds and was skipped."
                warnings.append(warning_message)
                self._log(logging.WARNING, f"Fallback usage | column={column_name} | reason=invalid_iqr_bounds")
                base_report.resolved_strategy = "ignore"
                base_report.fallback_used = True
                base_report.fallback_reason = "invalid_iqr_bounds"
                base_report.action_status = "skipped"
                return base_report, pd.Series(False, index=dataframe.index)

            outlier_mask = non_null_mask & ((converted_series < lower_bound) | (converted_series > upper_bound))
            outlier_count = int(outlier_mask.sum())
            outlier_ratio = self._safe_ratio(outlier_count, int(dataframe.shape[0]))
            outlier_ratio_non_null = self._safe_ratio(outlier_count, non_null_count)
            sample_rows = self._sample_outlier_rows(dataframe, column_name, outlier_mask, config)

            base_report.q1 = q1
            base_report.q3 = q3
            base_report.iqr = iqr
            base_report.lower_bound = lower_bound
            base_report.upper_bound = upper_bound
            base_report.outlier_count = outlier_count
            base_report.outlier_ratio = outlier_ratio
            base_report.outlier_ratio_non_null = outlier_ratio_non_null
            base_report.sample_rows = sample_rows
            base_report.requires_action = bool(outlier_count > 0 and base_report.resolved_strategy != "ignore")
            base_report.action_status = "pending" if outlier_count > 0 else "no_outliers"
            base_report.metadata["sample_values"] = schema_column.metadata.get("sample_values", [])
            base_report.metadata["iqr_multiplier"] = config.iqr_multiplier

            self._log(
                logging.INFO,
                f"Per-column outlier stats | column={column_name} | outliers={outlier_count} | "
                f"ratio={outlier_ratio:.4f} | lower={lower_bound:.6g} | upper={upper_bound:.6g}",
            )
            return base_report, outlier_mask
        except Exception as exc:
            warning_message = f"Column '{column_name}' failed outlier detection and was skipped: {exc}"
            warnings.append(warning_message)
            fallback_events.append(
                {
                    "column": column_name,
                    "original_strategy": config.column_strategy_overrides.get(column_name, config.outlier_strategy),
                    "fallback_strategy": "ignore",
                    "reason": str(exc),
                }
            )
            self._log(
                logging.WARNING,
                f"Fallback usage | column={column_name} | to=ignore | reason={type(exc).__name__}: {exc}",
            )
            return (
                OutlierColumnReport(
                    name=column_name,
                    inferred_type="numeric",
                    non_null_count=int(pd.to_numeric(dataframe[column_name], errors='coerce').notna().sum()),
                    outlier_count=0,
                    outlier_ratio=0.0,
                    outlier_ratio_non_null=0.0,
                    configured_strategy=str(config.column_strategy_overrides.get(column_name, config.outlier_strategy)),
                    resolved_strategy="ignore",
                    requires_action=False,
                    fallback_used=True,
                    fallback_reason=str(exc),
                    action_status="skipped",
                    metadata={"original_dtype": str(dataframe[column_name].dtype)},
                ),
                pd.Series(False, index=dataframe.index),
            )

    def _resolve_configured_strategy(
        self,
        column_name: str,
        config: OutlierHandlerConfig,
    ) -> str:
        try:
            raw_strategy = config.column_strategy_overrides.get(column_name, config.outlier_strategy)
            return self._canonicalize_strategy(raw_strategy)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | strategy resolution failed for column={column_name}: {exc}")
            raise OutlierHandlerStageError("Failed to resolve configured outlier strategy.", cause=exc) from exc

    def _resolve_fallback_strategy(
        self,
        column_name: str,
        config: OutlierHandlerConfig,
        warnings: List[str],
    ) -> str:
        try:
            try:
                return self._canonicalize_strategy(config.fallback_strategy)
            except Exception:
                warnings.append(
                    f"Column '{column_name}' could not use fallback strategy '{config.fallback_strategy}'. "
                    "Using safe ignore fallback instead."
                )
                return "ignore"
        except Exception as exc:
            self._log(logging.ERROR, f"Error | fallback strategy resolution failed for column={column_name}: {exc}")
            raise OutlierHandlerStageError("Failed to resolve fallback outlier strategy.", cause=exc) from exc

    def _canonicalize_strategy(self, strategy: str) -> str:
        try:
            normalized = str(strategy).strip().lower().replace("-", "_")
            if normalized in self._strategy_aliases:
                return self._strategy_aliases[normalized]
            raise ValueError(f"Unsupported outlier strategy '{strategy}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | strategy canonicalization failed for strategy={strategy}: {exc}")
            raise OutlierHandlerStageError("Failed to canonicalize outlier strategy.", cause=exc) from exc

    def _sample_outlier_rows(
        self,
        dataframe: pd.DataFrame,
        column_name: str,
        outlier_mask: pd.Series,
        config: OutlierHandlerConfig,
    ) -> List[Dict[str, Any]]:
        try:
            if not config.include_row_preview:
                return []

            sample_frame = dataframe.loc[outlier_mask]
            if sample_frame.empty:
                return []

            sample_frame = sample_frame.head(config.sample_row_count)
            if config.row_preview_column_limit is not None and config.row_preview_column_limit > 0:
                preview_columns = list(sample_frame.columns[: config.row_preview_column_limit])
            else:
                preview_columns = list(sample_frame.columns)

            sample_rows: List[Dict[str, Any]] = []
            for row_index, row in sample_frame.iterrows():
                serialized_row = {str(column): self._serialize_value(row[column]) for column in preview_columns}
                sample_rows.append(
                    {
                        "index": self._serialize_value(row_index),
                        "value": self._serialize_value(row[column_name]),
                        "row": serialized_row,
                    }
                )
            return sample_rows
        except Exception as exc:
            self._log(logging.ERROR, f"Error | outlier row sampling failed for column={column_name}: {exc}")
            raise OutlierHandlerStageError("Failed to sample outlier rows.", cause=exc) from exc

    def _build_strategy_summary(self, column_reports: Sequence[OutlierColumnReport]) -> Dict[str, List[str]]:
        try:
            summary: Dict[str, List[str]] = defaultdict(list)
            for column_report in column_reports:
                if column_report.outlier_count <= 0:
                    continue
                summary[column_report.resolved_strategy].append(column_report.name)
            return {strategy: sorted(columns) for strategy, columns in summary.items()}
        except Exception as exc:
            self._log(logging.ERROR, f"Error | strategy summary construction failed: {exc}")
            raise OutlierHandlerStageError("Failed to build outlier strategy summary.", cause=exc) from exc

    def _format_strategy_summary(self, summary: Dict[str, List[str]]) -> str:
        try:
            if not summary:
                return "no outlier actions required"
            return ", ".join(f"{strategy}={len(columns)}" for strategy, columns in sorted(summary.items()))
        except Exception as exc:
            self._log(logging.ERROR, f"Error | strategy summary formatting failed: {exc}")
            raise OutlierHandlerStageError("Failed to format outlier strategy summary.", cause=exc) from exc

    def _resolve_confirmation(
        self,
        report: OutlierHandlingReport,
        config: OutlierHandlerConfig,
    ) -> bool:
        try:
            if config.preview_only:
                report.confirmation_status = "preview_only"
                return False

            if not report.affected_columns:
                report.confirmation_status = "not_required"
                return True

            if not config.require_confirmation:
                report.confirmation_status = "not_required"
                return True

            if config.confirmation_callback is not None:
                try:
                    confirmed = bool(config.confirmation_callback(report))
                    report.confirmation_status = "confirmed" if confirmed else "rejected"
                    return confirmed
                except Exception as exc:
                    report.warnings.append(f"Confirmation callback failed: {exc}")
                    self._log(logging.WARNING, f"Fallback usage | confirmation callback failed: {exc}")
                    if config.auto_confirm:
                        report.confirmation_status = "simulated_confirmed"
                        return True
                    report.confirmation_status = "callback_failed"
                    return False

            if config.auto_confirm:
                report.confirmation_status = "simulated_confirmed"
                return True

            report.confirmation_status = "awaiting_confirmation"
            return False
        except Exception as exc:
            self._log(logging.ERROR, f"Error | confirmation resolution failed: {exc}")
            raise OutlierHandlerStageError("Failed to resolve outlier strategy confirmation.", cause=exc) from exc

    def _apply_strategies(
        self,
        dataframe: pd.DataFrame,
        report: OutlierHandlingReport,
        config: OutlierHandlerConfig,
    ) -> Tuple[pd.DataFrame, OutlierHandlingReport]:
        try:
            cleaned_frame = dataframe.copy(deep=True)
            rows_before = int(cleaned_frame.shape[0])

            group_columns = [
                column_report
                for column_report in report.column_reports
                if column_report.outlier_count > 0 and column_report.resolved_strategy == "group"
            ]
            if group_columns:
                flag_column_name = self._resolve_flag_column_name(cleaned_frame, config, report.warnings)
                if config.group_flag_mode == "label":
                    cleaned_frame[flag_column_name] = pd.NA
                else:
                    cleaned_frame[flag_column_name] = False

                for column_report in group_columns:
                    outlier_mask = self._mask_from_bounds(cleaned_frame, column_report)
                    if config.group_flag_mode == "label":
                        cleaned_frame.loc[outlier_mask, flag_column_name] = config.outlier_group_value
                    else:
                        cleaned_frame.loc[outlier_mask, flag_column_name] = True
                    column_report.applied_strategy = "group"
                    column_report.action_status = "grouped" if column_report.outlier_count > 0 else "no_outliers"

                report.flag_column_added = flag_column_name

            for column_report in report.column_reports:
                if column_report.outlier_count <= 0:
                    column_report.action_status = "no_outliers"
                    continue

                if column_report.resolved_strategy == "cap":
                    cleaned_frame[column_report.name] = self._cap_series(
                        series=cleaned_frame[column_report.name],
                        lower_bound=float(column_report.lower_bound),
                        upper_bound=float(column_report.upper_bound),
                    )
                    column_report.applied_strategy = "cap"
                    column_report.action_status = "capped"
                elif column_report.resolved_strategy == "ignore":
                    column_report.applied_strategy = "ignore"
                    column_report.action_status = "ignored"

            drop_columns = [
                column_report
                for column_report in report.column_reports
                if column_report.outlier_count > 0 and column_report.resolved_strategy == "drop"
            ]

            dropped_rows = 0
            if drop_columns:
                combined_drop_mask = pd.Series(False, index=cleaned_frame.index)
                for column_report in drop_columns:
                    combined_drop_mask = combined_drop_mask | self._mask_from_bounds(cleaned_frame, column_report)
                    column_report.applied_strategy = "drop"
                dropped_rows = int(combined_drop_mask.sum())
                cleaned_frame = cleaned_frame.loc[~combined_drop_mask].copy()
                if config.reset_index_after_drop:
                    cleaned_frame = cleaned_frame.reset_index(drop=True)
                for column_report in drop_columns:
                    column_report.action_status = "rows_dropped" if dropped_rows > 0 else "drop_strategy_no_rows"

            report.applied = True
            report.rows_after = int(cleaned_frame.shape[0])
            report.columns_after = int(cleaned_frame.shape[1])
            report.dropped_rows = dropped_rows

            self._log(logging.INFO, f"Strategy applied | {self._format_strategy_summary(report.strategy_summary)}")
            if report.flag_column_added is not None:
                self._log(logging.INFO, f"Strategy applied | group_flag_column={report.flag_column_added}")
            if dropped_rows > 0:
                self._log(logging.INFO, f"Strategy applied | dropped_rows={dropped_rows}")
            self._log(
                logging.INFO,
                f"Output summary | rows_before={rows_before} | rows_after={report.rows_after} | "
                f"dropped_rows={report.dropped_rows}",
            )

            return cleaned_frame, report
        except Exception as exc:
            self._log(logging.ERROR, f"Error | strategy application failed: {exc}")
            raise OutlierHandlerStageError("Failed to apply outlier strategies.", cause=exc) from exc

    def _mask_from_bounds(
        self,
        dataframe: pd.DataFrame,
        column_report: OutlierColumnReport,
    ) -> pd.Series:
        try:
            converted = pd.to_numeric(dataframe[column_report.name], errors="coerce")
            return converted.notna() & (
                (converted < float(column_report.lower_bound)) | (converted > float(column_report.upper_bound))
            )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | outlier mask reconstruction failed for column={column_report.name}: {exc}")
            raise OutlierHandlerStageError("Failed to reconstruct outlier mask.", cause=exc) from exc

    def _cap_series(
        self,
        series: pd.Series,
        lower_bound: float,
        upper_bound: float,
    ) -> pd.Series:
        try:
            converted = pd.to_numeric(series, errors="coerce")
            clipped = converted.clip(lower=lower_bound, upper=upper_bound)
            if is_numeric_dtype(series.dtype):
                return clipped
            result = series.copy()
            result.loc[converted.notna()] = clipped.loc[converted.notna()]
            return result
        except Exception as exc:
            self._log(logging.ERROR, f"Error | series capping failed: {exc}")
            raise OutlierHandlerStageError("Failed to cap outlier values.", cause=exc) from exc

    def _resolve_flag_column_name(
        self,
        dataframe: pd.DataFrame,
        config: OutlierHandlerConfig,
        warnings: List[str],
    ) -> str:
        try:
            base_name = config.outlier_flag_column
            if base_name not in dataframe.columns or config.overwrite_existing_flag_column:
                return base_name

            suffix = 1
            candidate = f"{base_name}__{suffix}"
            while candidate in dataframe.columns:
                suffix += 1
                candidate = f"{base_name}__{suffix}"
            warnings.append(
                f"Requested flag column '{base_name}' already exists. Using generated flag column '{candidate}'."
            )
            self._log(
                logging.WARNING,
                f"Fallback usage | requested_flag_column={base_name} | generated_flag_column={candidate}",
            )
            return candidate
        except Exception as exc:
            self._log(logging.ERROR, f"Error | flag column resolution failed: {exc}")
            raise OutlierHandlerStageError("Failed to resolve outlier flag column name.", cause=exc) from exc

    def _serialize_value(self, value: Any) -> Any:
        try:
            if pd.isna(value):
                return None
            if hasattr(value, "item"):
                return value.item()
            if hasattr(value, "isoformat"):
                return value.isoformat()
            return value
        except Exception:
            return str(value)

    def _safe_ratio(self, numerator: int, denominator: int) -> float:
        try:
            if denominator <= 0:
                return 0.0
            return float(numerator) / float(denominator)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | ratio computation failed: {exc}")
            raise OutlierHandlerStageError("Failed to compute outlier ratio.", cause=exc) from exc

    def _log(self, level: int, message: str) -> None:
        try:
            self.logger.log(level, f"[{self.stage_name}] - {message}")
        except Exception:
            logging.getLogger(__name__).log(level, f"[{self.stage_name}] - {message}")


def generate_outlier_report(
    dataframe: pd.DataFrame,
    schema: Any,
    config: Optional[OutlierHandlerConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> OutlierHandlingReport:
    try:
        module = OutlierHandlerModule(config=config, logger=logger)
        return module.generate_report(dataframe=dataframe, schema=schema)
    except OutlierHandlerStageError:
        raise
    except Exception as exc:
        raise OutlierHandlerStageError("Unhandled outlier report generation error.", cause=exc) from exc


def handle_outliers(
    dataframe: pd.DataFrame,
    schema: Any,
    config: Optional[OutlierHandlerConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> OutlierHandlerResult:
    try:
        module = OutlierHandlerModule(config=config, logger=logger)
        return module.run(dataframe=dataframe, schema=schema)
    except OutlierHandlerStageError:
        raise
    except Exception as exc:
        raise OutlierHandlerStageError("Unhandled outlier handler error.", cause=exc) from exc
