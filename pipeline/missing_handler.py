from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_categorical_dtype

from .schema_detection import DatasetSchema, SchemaColumn


ConfirmationCallback = Callable[["MissingHandlingReport"], bool]


@dataclass
class MissingHandlerConfig:
    missing_strategy: Dict[str, str] = field(
        default_factory=lambda: {
            "numeric": "mean",
            "categorical": "mode",
            "text": "empty",
            "unknown": "drop_rows",
        }
    )
    fallback_strategy: Dict[str, str] = field(
        default_factory=lambda: {
            "numeric": "drop_rows",
            "categorical": "unknown",
            "text": "unknown",
            "unknown": "drop_rows",
        }
    )
    column_strategy_overrides: Dict[str, str] = field(default_factory=dict)
    categorical_unknown_value: str = "unknown"
    text_unknown_value: str = "unknown"
    text_empty_value: str = ""
    numeric_zero_value: float = 0.0
    preview_only: bool = False
    require_confirmation: bool = True
    auto_confirm: bool = True
    confirmation_callback: Optional[ConfirmationCallback] = None
    reset_index_after_drop: bool = True
    include_columns_without_missing: bool = True


@dataclass
class MissingColumnReport:
    name: str
    inferred_type: str
    missing_count: int
    missing_ratio: float
    configured_strategy: str
    resolved_strategy: str
    requires_action: bool
    applied_strategy: Optional[str] = None
    fill_value_preview: Any = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    action_status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MissingHandlingReport:
    stage: str
    rows_before: int
    columns_before: int
    total_missing_cells: int
    affected_columns: List[str]
    column_reports: List[MissingColumnReport] = field(default_factory=list)
    strategy_summary: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    confirmation_required: bool = True
    confirmation_status: str = "pending"
    preview_generated: bool = True
    applied: bool = False
    rows_after: Optional[int] = None
    columns_after: Optional[int] = None
    dropped_rows: int = 0
    fallback_events: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class MissingHandlerResult:
    dataframe: pd.DataFrame
    report: MissingHandlingReport


class MissingHandlerStageError(RuntimeError):
    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        try:
            super().__init__(message)
            self.cause = cause
        except Exception as exc:
            raise RuntimeError("Failed to initialize MissingHandlerStageError.") from exc


class MissingHandlerModule:
    stage_name = "MISSING_HANDLER"
    _supported_types = {"numeric", "categorical", "text", "unknown"}
    _strategy_aliases = {
        "numeric": {
            "mean": "mean",
            "median": "median",
            "zero": "zero",
            "zero_fill": "zero",
            "zero fill": "zero",
            "drop_rows": "drop_rows",
            "drop rows": "drop_rows",
        },
        "categorical": {
            "mode": "mode",
            "unknown": "unknown",
            "unknown_fill": "unknown",
            "unknown fill": "unknown",
            "drop_rows": "drop_rows",
            "drop rows": "drop_rows",
        },
        "text": {
            "empty": "empty",
            "empty_string": "empty",
            "empty string": "empty",
            "empty_string_fill": "empty",
            "empty string fill": "empty",
            "unknown": "unknown",
            "unknown_token": "unknown",
            "unknown token": "unknown",
            "drop_rows": "drop_rows",
            "drop rows": "drop_rows",
        },
        "unknown": {
            "unknown": "unknown",
            "unknown_fill": "unknown",
            "unknown fill": "unknown",
            "drop_rows": "drop_rows",
            "drop rows": "drop_rows",
        },
    }

    def __init__(
        self,
        config: Optional[MissingHandlerConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        try:
            self.config = config or MissingHandlerConfig()
            self.logger = logger or logging.getLogger(__name__)
        except Exception as exc:
            raise MissingHandlerStageError("Failed to initialize missing handler module.", cause=exc) from exc

    def generate_report(
        self,
        dataframe: pd.DataFrame,
        schema: Any,
        config: Optional[MissingHandlerConfig] = None,
    ) -> MissingHandlingReport:
        try:
            active_config = config or self.config
            validated_frame, schema_lookup, warnings = self._validate_inputs(dataframe, schema)
            self._log(
                logging.INFO,
                f"Stage start | rows={validated_frame.shape[0]} | columns={validated_frame.shape[1]}",
            )

            total_missing_cells = int(validated_frame.isna().sum().sum())
            affected_columns: List[str] = []
            column_reports: List[MissingColumnReport] = []
            fallback_events: List[Dict[str, Any]] = []

            for column_name in validated_frame.columns:
                column_report = self._build_column_report(
                    series=validated_frame[column_name],
                    column_name=str(column_name),
                    schema_column=schema_lookup.get(str(column_name)),
                    config=active_config,
                    warnings=warnings,
                    fallback_events=fallback_events,
                )

                if column_report.requires_action:
                    affected_columns.append(column_report.name)

                if active_config.include_columns_without_missing or column_report.requires_action:
                    column_reports.append(column_report)

            strategy_summary = self._build_strategy_summary(column_reports)
            self._log(
                logging.INFO,
                f"Missing distribution summary | total_missing_cells={total_missing_cells} | "
                f"affected_columns={len(affected_columns)}",
            )

            if fallback_events:
                self._log(logging.WARNING, f"Fallback usage | count={len(fallback_events)}")

            report = MissingHandlingReport(
                stage=self.stage_name,
                rows_before=int(validated_frame.shape[0]),
                columns_before=int(validated_frame.shape[1]),
                total_missing_cells=total_missing_cells,
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

            self._log(
                logging.INFO,
                f"Strategy preview | {self._format_strategy_summary(strategy_summary)}",
            )

            return report
        except MissingHandlerStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | report generation failed with {type(exc).__name__}: {exc}")
            raise MissingHandlerStageError("Missing value report generation failed.", cause=exc) from exc

    def run(
        self,
        dataframe: pd.DataFrame,
        schema: Any,
        config: Optional[MissingHandlerConfig] = None,
    ) -> MissingHandlerResult:
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
                    if column_report.requires_action:
                        column_report.action_status = (
                            "preview_only" if active_config.preview_only else "awaiting_confirmation"
                        )
                    else:
                        column_report.action_status = "no_missing"
                return MissingHandlerResult(dataframe=preview_frame, report=report)

            cleaned_frame, updated_report = self._apply_strategies(
                dataframe=dataframe,
                report=report,
                config=active_config,
            )
            return MissingHandlerResult(dataframe=cleaned_frame, report=updated_report)
        except MissingHandlerStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage failed with {type(exc).__name__}: {exc}")
            raise MissingHandlerStageError("Missing handling failed.", cause=exc) from exc

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

            dataframe_column_names = {str(column) for column in dataframe.columns}
            for column_name in dataframe.columns:
                if str(column_name) not in schema_lookup:
                    warnings.append(
                        f"Column '{column_name}' is missing from schema; using fallback inferred type 'unknown'."
                    )
                    schema_lookup[str(column_name)] = SchemaColumn(
                        name=str(column_name),
                        inferred_type="unknown",
                        metadata={},
                    )

            for schema_name in schema_lookup:
                if schema_name not in dataframe_column_names:
                    warnings.append(f"Schema column '{schema_name}' is not present in the DataFrame.")

            return dataframe, schema_lookup, warnings
        except Exception as exc:
            self._log(logging.ERROR, f"Error | input validation failed: {exc}")
            raise MissingHandlerStageError("Invalid input for missing handler.", cause=exc) from exc

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
            raise MissingHandlerStageError("Failed to extract schema columns.", cause=exc) from exc

    def _build_column_report(
        self,
        series: pd.Series,
        column_name: str,
        schema_column: Optional[SchemaColumn],
        config: MissingHandlerConfig,
        warnings: List[str],
        fallback_events: List[Dict[str, Any]],
    ) -> MissingColumnReport:
        inferred_type = "unknown"
        try:
            missing_count = int(series.isna().sum())
            missing_ratio = self._safe_ratio(missing_count, int(series.shape[0]))
            inferred_type = self._normalize_inferred_type(
                schema_column.inferred_type if schema_column is not None else "unknown",
                column_name=column_name,
                warnings=warnings,
            )
            raw_configured_strategy = config.column_strategy_overrides.get(
                column_name,
                config.missing_strategy.get(inferred_type, config.fallback_strategy.get(inferred_type, "drop_rows")),
            )
            try:
                configured_strategy = self._resolve_configured_strategy(column_name, inferred_type, config)
                fallback_used = False
                fallback_reason: Optional[str] = None
            except Exception as exc:
                warnings.append(
                    f"Column '{column_name}' has invalid configured strategy for type '{inferred_type}': {exc}"
                )
                configured_strategy = str(raw_configured_strategy)
                resolved_strategy = self._resolve_fallback_strategy(
                    inferred_type=inferred_type,
                    original_strategy="invalid",
                    column_name=column_name,
                    config=config,
                    warnings=warnings,
                )
                fallback_events.append(
                    {
                            "column": column_name,
                            "inferred_type": inferred_type,
                            "original_strategy": "invalid",
                            "fallback_strategy": resolved_strategy,
                            "reason": str(exc),
                        }
                    )
                self._log(
                    logging.WARNING,
                    f"Fallback usage | column={column_name} | from=invalid | to={resolved_strategy} | "
                    f"reason={exc}",
                )
                fallback_used = True
                fallback_reason = str(exc)
            else:
                resolved_strategy = configured_strategy

            fill_value_preview = None

            if missing_count > 0 and resolved_strategy != "drop_rows":
                try:
                    fill_value_preview = self._preview_fill_value(series, inferred_type, resolved_strategy, config)
                except Exception as exc:
                    fallback_used = True
                    fallback_reason = str(exc)
                    resolved_strategy = self._resolve_fallback_strategy(
                        inferred_type=inferred_type,
                        original_strategy=configured_strategy,
                        column_name=column_name,
                        config=config,
                        warnings=warnings,
                    )
                    fallback_events.append(
                        {
                            "column": column_name,
                            "inferred_type": inferred_type,
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
                    if resolved_strategy != "drop_rows":
                        fill_value_preview = self._preview_fill_value(series, inferred_type, resolved_strategy, config)

            metadata = {
                "original_dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "schema_confidence": self._extract_schema_confidence(schema_column),
                "sample_values": self._extract_schema_samples(schema_column),
                "available_strategies": self._available_strategies(inferred_type),
            }

            action_status = "pending" if missing_count > 0 else "no_missing"
            return MissingColumnReport(
                name=column_name,
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                configured_strategy=configured_strategy,
                resolved_strategy=resolved_strategy,
                requires_action=bool(missing_count > 0),
                fill_value_preview=fill_value_preview,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                action_status=action_status,
                metadata=metadata,
            )
        except Exception as exc:
            warnings.append(f"Column '{column_name}' failed missing strategy planning: {exc}")
            fallback_strategy = self._resolve_fallback_strategy(
                inferred_type=inferred_type,
                original_strategy="unknown",
                column_name=column_name,
                config=config,
                warnings=warnings,
            )
            fallback_events.append(
                {
                    "column": column_name,
                    "inferred_type": "unknown",
                    "original_strategy": "unknown",
                    "fallback_strategy": fallback_strategy,
                    "reason": str(exc),
                }
            )
            self._log(
                logging.WARNING,
                f"Fallback usage | column={column_name} | from=unknown | to={fallback_strategy} | "
                f"reason={type(exc).__name__}: {exc}",
            )
            missing_count = int(series.isna().sum())
            return MissingColumnReport(
                name=column_name,
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_ratio=self._safe_ratio(missing_count, int(series.shape[0])),
                configured_strategy="unknown",
                resolved_strategy=fallback_strategy,
                requires_action=bool(missing_count > 0),
                fallback_used=True,
                fallback_reason=str(exc),
                action_status="pending" if missing_count > 0 else "no_missing",
                metadata={
                    "original_dtype": str(series.dtype),
                    "available_strategies": self._available_strategies(inferred_type),
                },
            )

    def _normalize_inferred_type(
        self,
        inferred_type: str,
        column_name: str,
        warnings: List[str],
    ) -> str:
        try:
            normalized = str(inferred_type).strip().lower()
            if normalized not in self._supported_types:
                warnings.append(
                    f"Column '{column_name}' has unsupported schema type '{inferred_type}'; using 'unknown'."
                )
                return "unknown"
            return normalized
        except Exception as exc:
            self._log(logging.ERROR, f"Error | inferred type normalization failed for column={column_name}: {exc}")
            raise MissingHandlerStageError("Failed to normalize inferred column type.", cause=exc) from exc

    def _resolve_configured_strategy(
        self,
        column_name: str,
        inferred_type: str,
        config: MissingHandlerConfig,
    ) -> str:
        try:
            raw_strategy = config.column_strategy_overrides.get(
                column_name,
                config.missing_strategy.get(inferred_type, config.fallback_strategy.get(inferred_type, "drop_rows")),
            )
            return self._canonicalize_strategy(raw_strategy, inferred_type)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | strategy resolution failed for column={column_name}: {exc}")
            raise MissingHandlerStageError("Failed to resolve configured missing strategy.", cause=exc) from exc

    def _resolve_fallback_strategy(
        self,
        inferred_type: str,
        original_strategy: str,
        column_name: str,
        config: MissingHandlerConfig,
        warnings: List[str],
    ) -> str:
        try:
            fallback_raw = config.fallback_strategy.get(inferred_type, "drop_rows")
            try:
                return self._canonicalize_strategy(fallback_raw, inferred_type)
            except Exception:
                warnings.append(
                    f"Column '{column_name}' could not use fallback strategy '{fallback_raw}' for type "
                    f"'{inferred_type}'. Using safe drop_rows fallback instead."
                )
                return self._canonicalize_strategy("drop_rows", "unknown" if inferred_type == "unknown" else inferred_type)
        except Exception as exc:
            self._log(
                logging.ERROR,
                f"Error | fallback strategy resolution failed for column={column_name}, strategy={original_strategy}: {exc}",
            )
            raise MissingHandlerStageError("Failed to resolve fallback missing strategy.", cause=exc) from exc

    def _canonicalize_strategy(self, strategy: str, inferred_type: str) -> str:
        try:
            alias_map = self._strategy_aliases.get(inferred_type, self._strategy_aliases["unknown"])
            normalized = str(strategy).strip().lower().replace("-", "_")
            if normalized in alias_map:
                return alias_map[normalized]
            raise ValueError(f"Unsupported strategy '{strategy}' for inferred type '{inferred_type}'.")
        except Exception as exc:
            self._log(
                logging.ERROR,
                f"Error | strategy canonicalization failed for strategy={strategy}, inferred_type={inferred_type}: {exc}",
            )
            raise MissingHandlerStageError("Failed to canonicalize missing strategy.", cause=exc) from exc

    def _available_strategies(self, inferred_type: str) -> List[str]:
        try:
            alias_map = self._strategy_aliases.get(inferred_type, self._strategy_aliases["unknown"])
            return sorted(set(alias_map.values()))
        except Exception as exc:
            self._log(logging.ERROR, f"Error | available strategy lookup failed for type={inferred_type}: {exc}")
            raise MissingHandlerStageError("Failed to list available strategies.", cause=exc) from exc

    def _preview_fill_value(
        self,
        series: pd.Series,
        inferred_type: str,
        strategy: str,
        config: MissingHandlerConfig,
    ) -> Any:
        try:
            if strategy == "drop_rows":
                return None

            if inferred_type == "numeric":
                return self._numeric_fill_value(series, strategy, config)
            if inferred_type == "categorical":
                return self._categorical_fill_value(series, strategy, config)
            if inferred_type == "text":
                return self._text_fill_value(strategy, config)
            if inferred_type == "unknown":
                return self._unknown_fill_value(strategy, config)

            raise ValueError(f"Unsupported inferred type '{inferred_type}'.")
        except Exception as exc:
            self._log(
                logging.ERROR,
                f"Error | fill value preview failed for column dtype={series.dtype}, type={inferred_type}, "
                f"strategy={strategy}: {exc}",
            )
            raise MissingHandlerStageError("Failed to preview missing fill value.", cause=exc) from exc

    def _numeric_fill_value(self, series: pd.Series, strategy: str, config: MissingHandlerConfig) -> Any:
        try:
            if strategy == "zero":
                return config.numeric_zero_value

            numeric_series = pd.to_numeric(series.dropna(), errors="coerce").dropna()
            if numeric_series.empty:
                raise ValueError("Numeric fill requires at least one non-null numeric value.")

            if strategy == "mean":
                return float(numeric_series.mean())
            if strategy == "median":
                return float(numeric_series.median())

            raise ValueError(f"Unsupported numeric strategy '{strategy}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | numeric fill value computation failed: {exc}")
            raise MissingHandlerStageError("Failed to compute numeric fill value.", cause=exc) from exc

    def _categorical_fill_value(self, series: pd.Series, strategy: str, config: MissingHandlerConfig) -> Any:
        try:
            if strategy == "unknown":
                return config.categorical_unknown_value

            if strategy == "mode":
                mode_values = series.dropna().mode(dropna=True)
                if mode_values.empty:
                    raise ValueError("Categorical mode fill requires at least one non-null value.")
                return self._serialize_fill_value(mode_values.iloc[0])

            raise ValueError(f"Unsupported categorical strategy '{strategy}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | categorical fill value computation failed: {exc}")
            raise MissingHandlerStageError("Failed to compute categorical fill value.", cause=exc) from exc

    def _text_fill_value(self, strategy: str, config: MissingHandlerConfig) -> Any:
        try:
            if strategy == "empty":
                return config.text_empty_value
            if strategy == "unknown":
                return config.text_unknown_value
            raise ValueError(f"Unsupported text strategy '{strategy}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | text fill value computation failed: {exc}")
            raise MissingHandlerStageError("Failed to compute text fill value.", cause=exc) from exc

    def _unknown_fill_value(self, strategy: str, config: MissingHandlerConfig) -> Any:
        try:
            if strategy == "unknown":
                return config.categorical_unknown_value
            raise ValueError(f"Unsupported unknown strategy '{strategy}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | unknown fill value computation failed: {exc}")
            raise MissingHandlerStageError("Failed to compute unknown fill value.", cause=exc) from exc

    def _serialize_fill_value(self, value: Any) -> Any:
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

    def _extract_schema_confidence(self, schema_column: Optional[SchemaColumn]) -> Optional[float]:
        try:
            if schema_column is None:
                return None
            confidence = schema_column.metadata.get("inference_confidence")
            return float(confidence) if confidence is not None else None
        except Exception:
            return None

    def _extract_schema_samples(self, schema_column: Optional[SchemaColumn]) -> List[Any]:
        try:
            if schema_column is None:
                return []
            raw_values = schema_column.metadata.get("sample_values", [])
            return raw_values if isinstance(raw_values, list) else []
        except Exception:
            return []

    def _build_strategy_summary(self, column_reports: Sequence[MissingColumnReport]) -> Dict[str, Dict[str, List[str]]]:
        try:
            summary: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
            for column_report in column_reports:
                if not column_report.requires_action:
                    continue
                summary[column_report.inferred_type][column_report.resolved_strategy].append(column_report.name)

            formatted_summary: Dict[str, Dict[str, List[str]]] = {}
            for inferred_type, strategy_map in summary.items():
                formatted_summary[inferred_type] = {}
                for strategy_name, column_names in strategy_map.items():
                    formatted_summary[inferred_type][strategy_name] = sorted(column_names)
            return formatted_summary
        except Exception as exc:
            self._log(logging.ERROR, f"Error | strategy summary construction failed: {exc}")
            raise MissingHandlerStageError("Failed to build missing strategy summary.", cause=exc) from exc

    def _format_strategy_summary(self, summary: Dict[str, Dict[str, List[str]]]) -> str:
        try:
            if not summary:
                return "no missing actions required"

            segments: List[str] = []
            for inferred_type in sorted(summary):
                for strategy_name in sorted(summary[inferred_type]):
                    segments.append(f"{inferred_type}:{strategy_name}={len(summary[inferred_type][strategy_name])}")
            return ", ".join(segments)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | strategy summary formatting failed: {exc}")
            raise MissingHandlerStageError("Failed to format missing strategy summary.", cause=exc) from exc

    def _resolve_confirmation(
        self,
        report: MissingHandlingReport,
        config: MissingHandlerConfig,
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
            raise MissingHandlerStageError("Failed to resolve missing strategy confirmation.", cause=exc) from exc

    def _apply_strategies(
        self,
        dataframe: pd.DataFrame,
        report: MissingHandlingReport,
        config: MissingHandlerConfig,
    ) -> Tuple[pd.DataFrame, MissingHandlingReport]:
        try:
            cleaned_frame = dataframe.copy(deep=True)
            rows_before = int(cleaned_frame.shape[0])

            drop_columns = [
                column_report.name
                for column_report in report.column_reports
                if column_report.requires_action and column_report.resolved_strategy == "drop_rows"
            ]

            dropped_rows = 0
            if drop_columns:
                missing_mask = cleaned_frame[drop_columns].isna().any(axis=1)
                dropped_rows = int(missing_mask.sum())
                cleaned_frame = cleaned_frame.loc[~missing_mask].copy()
                if config.reset_index_after_drop:
                    cleaned_frame = cleaned_frame.reset_index(drop=True)

            for column_report in report.column_reports:
                if not column_report.requires_action:
                    column_report.action_status = "no_missing"
                    continue

                if column_report.resolved_strategy == "drop_rows":
                    column_report.applied_strategy = "drop_rows"
                    column_report.action_status = "rows_dropped" if dropped_rows > 0 else "drop_strategy_no_rows"
                    continue

                fill_value = column_report.fill_value_preview
                if fill_value is None:
                    fill_value = self._preview_fill_value(
                        series=cleaned_frame[column_report.name],
                        inferred_type=column_report.inferred_type,
                        strategy=column_report.resolved_strategy,
                        config=config,
                    )

                cleaned_frame[column_report.name] = self._fill_series(
                    series=cleaned_frame[column_report.name],
                    fill_value=fill_value,
                )
                column_report.applied_strategy = column_report.resolved_strategy
                column_report.action_status = "filled"

            report.applied = True
            report.rows_after = int(cleaned_frame.shape[0])
            report.columns_after = int(cleaned_frame.shape[1])
            report.dropped_rows = dropped_rows

            self._log(
                logging.INFO,
                f"Strategy applied per column group | {self._format_strategy_summary(report.strategy_summary)}",
            )
            if dropped_rows > 0:
                self._log(logging.INFO, f"Strategy applied per column group | dropped_rows={dropped_rows}")

            self._log(
                logging.INFO,
                f"Output summary | rows_before={rows_before} | rows_after={report.rows_after} | "
                f"dropped_rows={report.dropped_rows}",
            )

            return cleaned_frame, report
        except Exception as exc:
            self._log(logging.ERROR, f"Error | strategy application failed: {exc}")
            raise MissingHandlerStageError("Failed to apply missing value strategies.", cause=exc) from exc

    def _fill_series(self, series: pd.Series, fill_value: Any) -> pd.Series:
        try:
            if is_categorical_dtype(series.dtype):
                categorical_series = series
                if fill_value not in categorical_series.cat.categories:
                    categorical_series = categorical_series.cat.add_categories([fill_value])
                return categorical_series.fillna(fill_value)
            return series.fillna(fill_value)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | series fill failed: {exc}")
            raise MissingHandlerStageError("Failed to fill missing values in series.", cause=exc) from exc

    def _safe_ratio(self, numerator: int, denominator: int) -> float:
        try:
            if denominator <= 0:
                return 0.0
            return float(numerator) / float(denominator)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | ratio computation failed: {exc}")
            raise MissingHandlerStageError("Failed to compute missing ratio.", cause=exc) from exc

    def _log(self, level: int, message: str) -> None:
        try:
            self.logger.log(level, f"[{self.stage_name}] - {message}")
        except Exception:
            logging.getLogger(__name__).log(level, f"[{self.stage_name}] - {message}")


def generate_missing_report(
    dataframe: pd.DataFrame,
    schema: Any,
    config: Optional[MissingHandlerConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> MissingHandlingReport:
    try:
        module = MissingHandlerModule(config=config, logger=logger)
        return module.generate_report(dataframe=dataframe, schema=schema)
    except MissingHandlerStageError:
        raise
    except Exception as exc:
        raise MissingHandlerStageError("Unhandled missing report generation error.", cause=exc) from exc


def handle_missing_values(
    dataframe: pd.DataFrame,
    schema: Any,
    config: Optional[MissingHandlerConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> MissingHandlerResult:
    try:
        module = MissingHandlerModule(config=config, logger=logger)
        return module.run(dataframe=dataframe, schema=schema)
    except MissingHandlerStageError:
        raise
    except Exception as exc:
        raise MissingHandlerStageError("Unhandled missing handler error.", cause=exc) from exc
