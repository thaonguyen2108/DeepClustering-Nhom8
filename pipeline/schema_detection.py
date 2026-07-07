from __future__ import annotations

import logging
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


@dataclass
class SchemaDetectionConfig:
    analysis_sample_size: int = 5000
    sample_value_count: int = 5
    numeric_inference_threshold: float = 0.95
    datetime_inference_threshold: float = 0.95
    categorical_unique_ratio_threshold: float = 0.05
    categorical_unique_count_threshold: int = 50
    text_unique_ratio_threshold: float = 0.30
    text_avg_length_threshold: float = 25.0
    text_avg_token_count_threshold: float = 3.0
    text_long_value_ratio_threshold: float = 0.30
    text_long_value_length_threshold: int = 20
    text_score_threshold: float = 0.55
    unknown_confidence: float = 0.0
    include_sample_values: bool = True
    include_distribution_summary: bool = True
    unique_ratio_denominator: str = "non_null"
    treat_datetime_as_categorical: bool = True
    enable_datetime_inference: bool = True
    allow_numeric_string_inference: bool = True


@dataclass
class SchemaColumn:
    name: str
    inferred_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSchema:
    stage: str
    columns: List[SchemaColumn]
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class SchemaDetectionStageError(RuntimeError):
    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        try:
            super().__init__(message)
            self.cause = cause
        except Exception as exc:
            raise RuntimeError("Failed to initialize SchemaDetectionStageError.") from exc


class SchemaDetectionModule:
    stage_name = "SCHEMA_DETECTION"
    _supported_types = {"numeric", "categorical", "text", "unknown"}

    def __init__(
        self,
        config: Optional[SchemaDetectionConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        try:
            self.config = config or SchemaDetectionConfig()
            self.logger = logger or logging.getLogger(__name__)
        except Exception as exc:
            raise SchemaDetectionStageError("Failed to initialize schema detection module.", cause=exc) from exc

    def run(
        self,
        dataframe: pd.DataFrame,
        config: Optional[SchemaDetectionConfig] = None,
        upstream_metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetSchema:
        try:
            active_config = config or self.config
            validated_frame = self._validate_dataframe(dataframe)
            self._log(
                logging.INFO,
                f"Stage start | rows={validated_frame.shape[0]} | columns={validated_frame.shape[1]}",
            )

            warnings: List[str] = []
            schema_columns: List[SchemaColumn] = []

            for column_index, column_name in enumerate(validated_frame.columns):
                schema_columns.append(
                    self._analyze_column(
                        dataframe=validated_frame,
                        column_name=column_name,
                        column_index=column_index,
                        config=active_config,
                        warnings=warnings,
                    )
                )

            type_distribution = self._summarize_types(schema_columns)
            metadata = self._build_schema_metadata(
                dataframe=validated_frame,
                schema_columns=schema_columns,
                type_distribution=type_distribution,
                config=active_config,
                upstream_metadata=upstream_metadata,
                warnings=warnings,
            )

            self._log(logging.INFO, f"Columns processed | count={len(schema_columns)}")
            self._log(logging.INFO, f"Type distribution summary | {self._format_distribution(type_distribution)}")

            if warnings:
                self._log(logging.WARNING, f"Inference issues | count={len(warnings)}")

            return DatasetSchema(
                stage=self.stage_name,
                columns=schema_columns,
                metadata=metadata,
                warnings=warnings,
            )
        except SchemaDetectionStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage failed with {type(exc).__name__}: {exc}")
            raise SchemaDetectionStageError("Schema detection failed.", cause=exc) from exc

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(f"Expected pandas DataFrame, received {type(dataframe).__name__}.")
            if dataframe.shape[1] == 0:
                raise ValueError("Input DataFrame does not contain any columns.")
            return dataframe
        except Exception as exc:
            self._log(logging.ERROR, f"Error | dataframe validation failed: {exc}")
            raise SchemaDetectionStageError("Invalid input for schema detection.", cause=exc) from exc

    def _analyze_column(
        self,
        dataframe: pd.DataFrame,
        column_name: Any,
        column_index: int,
        config: SchemaDetectionConfig,
        warnings: List[str],
    ) -> SchemaColumn:
        try:
            series = dataframe.iloc[:, column_index]
            base_metadata = self._compute_base_metadata(series, column_name, column_index, config)
            inferred_type, inference_metadata = self._infer_column_type(series, base_metadata, config)
            column_metadata = {**base_metadata, **inference_metadata}

            if inferred_type not in self._supported_types:
                warnings.append(
                    f"Column '{column_name}' produced unsupported inferred type '{inferred_type}', falling back to unknown."
                )
                self._log(
                    logging.WARNING,
                    f"Inference issue | column={column_name} | unsupported_type={inferred_type} | fallback=unknown",
                )
                inferred_type = "unknown"
                column_metadata["inference_source"] = "fallback_unknown"
                column_metadata["inference_confidence"] = config.unknown_confidence

            return SchemaColumn(name=str(column_name), inferred_type=inferred_type, metadata=column_metadata)
        except Exception as exc:
            warnings.append(f"Column '{column_name}' failed schema inference: {exc}")
            self._log(
                logging.WARNING,
                f"Inference issue | column={column_name} | error={type(exc).__name__}: {exc} | fallback=unknown",
            )
            return SchemaColumn(
                name=str(column_name),
                inferred_type="unknown",
                metadata={
                    "column_index": int(column_index),
                    "original_dtype": str(dataframe.iloc[:, column_index].dtype),
                    "inference_source": "fallback_unknown",
                    "inference_confidence": config.unknown_confidence,
                    "error": str(exc),
                },
            )

    def _compute_base_metadata(
        self,
        series: pd.Series,
        column_name: Any,
        column_index: int,
        config: SchemaDetectionConfig,
    ) -> Dict[str, Any]:
        try:
            total_count = int(series.shape[0])
            non_null_series = series.dropna()
            non_null_count = int(non_null_series.shape[0])
            missing_count = total_count - non_null_count
            missing_ratio = self._safe_ratio(missing_count, total_count)
            unique_count = int(non_null_series.nunique(dropna=True))
            denominator = non_null_count if config.unique_ratio_denominator == "non_null" else total_count
            unique_ratio = self._safe_ratio(unique_count, denominator)
            sampled_values = self._sample_series(non_null_series, config.analysis_sample_size)
            string_metrics = self._compute_string_metrics(sampled_values, config)

            metadata: Dict[str, Any] = {
                "column_index": int(column_index),
                "original_name": str(column_name),
                "original_dtype": str(series.dtype),
                "row_count": total_count,
                "non_null_count": non_null_count,
                "missing_count": int(missing_count),
                "missing_ratio": missing_ratio,
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "avg_string_length": string_metrics["avg_length"],
                "avg_token_count": string_metrics["avg_token_count"],
                "long_value_ratio": string_metrics["long_value_ratio"],
                "max_string_length": string_metrics["max_length"],
            }

            if config.include_sample_values:
                metadata["sample_values"] = self._sample_values(non_null_series, config.sample_value_count)

            return metadata
        except Exception as exc:
            self._log(logging.ERROR, f"Error | base metadata computation failed for column={column_name}: {exc}")
            raise SchemaDetectionStageError("Failed to compute base schema metadata.", cause=exc) from exc

    def _infer_column_type(
        self,
        series: pd.Series,
        base_metadata: Dict[str, Any],
        config: SchemaDetectionConfig,
    ) -> tuple[str, Dict[str, Any]]:
        try:
            non_null_series = series.dropna()
            if non_null_series.empty:
                return (
                    "unknown",
                    {
                        "inference_source": "empty_column",
                        "inference_confidence": config.unknown_confidence,
                    },
                )

            if is_bool_dtype(series.dtype):
                return (
                    "categorical",
                    {
                        "inference_source": "boolean_dtype",
                        "inference_confidence": 0.99,
                        "logical_subtype": "boolean",
                    },
                )

            if is_categorical_dtype(series.dtype):
                return (
                    "categorical",
                    {
                        "inference_source": "native_categorical_dtype",
                        "inference_confidence": 0.99,
                        "logical_subtype": "categorical",
                    },
                )

            if is_numeric_dtype(series.dtype):
                return (
                    "numeric",
                    {
                        "inference_source": "native_numeric_dtype",
                        "inference_confidence": 0.99,
                        "logical_subtype": "numeric",
                    },
                )

            if is_datetime64_any_dtype(series.dtype):
                inferred_type = "categorical" if config.treat_datetime_as_categorical else "unknown"
                confidence = 0.90 if inferred_type == "categorical" else config.unknown_confidence
                return (
                    inferred_type,
                    {
                        "inference_source": "native_datetime_dtype",
                        "inference_confidence": confidence,
                        "logical_subtype": "datetime",
                    },
                )

            numeric_parse_ratio = 0.0
            if config.allow_numeric_string_inference:
                numeric_parse_ratio = self._numeric_parse_ratio(non_null_series, config)
                if numeric_parse_ratio >= config.numeric_inference_threshold:
                    return (
                        "numeric",
                        {
                            "inference_source": "numeric_string_inference",
                            "inference_confidence": self._bounded_confidence(numeric_parse_ratio),
                            "logical_subtype": "numeric_string",
                            "numeric_parse_ratio": numeric_parse_ratio,
                        },
                    )

            datetime_parse_ratio = 0.0
            if config.enable_datetime_inference:
                datetime_parse_ratio = self._datetime_parse_ratio(non_null_series, config)
                if datetime_parse_ratio >= config.datetime_inference_threshold:
                    inferred_type = "categorical" if config.treat_datetime_as_categorical else "unknown"
                    confidence = self._bounded_confidence(datetime_parse_ratio if inferred_type == "categorical" else 0.0)
                    return (
                        inferred_type,
                        {
                            "inference_source": "datetime_string_inference",
                            "inference_confidence": confidence,
                            "logical_subtype": "datetime_string",
                            "datetime_parse_ratio": datetime_parse_ratio,
                        },
                    )

            text_score = self._text_score(base_metadata, config)
            if text_score >= config.text_score_threshold:
                return (
                    "text",
                    {
                        "inference_source": "text_heuristics",
                        "inference_confidence": self._bounded_confidence(text_score),
                        "text_score": text_score,
                        "numeric_parse_ratio": numeric_parse_ratio,
                        "datetime_parse_ratio": datetime_parse_ratio,
                    },
                )

            categorical_confidence = self._categorical_confidence(base_metadata, config)
            return (
                "categorical",
                {
                    "inference_source": "categorical_fallback",
                    "inference_confidence": categorical_confidence,
                    "text_score": text_score,
                    "numeric_parse_ratio": numeric_parse_ratio,
                    "datetime_parse_ratio": datetime_parse_ratio,
                },
            )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | type inference failed for column={base_metadata.get('original_name')}: {exc}")
            raise SchemaDetectionStageError("Failed to infer column type.", cause=exc) from exc

    def _sample_series(self, series: pd.Series, sample_size: int) -> pd.Series:
        try:
            if sample_size <= 0 or series.shape[0] <= sample_size:
                return series
            return series.head(sample_size)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | series sampling failed: {exc}")
            raise SchemaDetectionStageError("Failed to sample series for schema detection.", cause=exc) from exc

    def _compute_string_metrics(
        self,
        series: pd.Series,
        config: SchemaDetectionConfig,
    ) -> Dict[str, float]:
        try:
            if series.empty:
                return {
                    "avg_length": 0.0,
                    "avg_token_count": 0.0,
                    "long_value_ratio": 0.0,
                    "max_length": 0.0,
                }

            normalized_strings = series.map(self._safe_to_string)
            lengths = normalized_strings.str.len()
            token_counts = normalized_strings.map(lambda value: len(value.split()))
            long_value_ratio = self._safe_ratio(
                int((lengths >= config.text_long_value_length_threshold).sum()),
                int(lengths.shape[0]),
            )

            return {
                "avg_length": float(lengths.mean()) if not lengths.empty else 0.0,
                "avg_token_count": float(token_counts.mean()) if not token_counts.empty else 0.0,
                "long_value_ratio": long_value_ratio,
                "max_length": float(lengths.max()) if not lengths.empty else 0.0,
            }
        except Exception as exc:
            self._log(logging.ERROR, f"Error | string metric computation failed: {exc}")
            raise SchemaDetectionStageError("Failed to compute string metrics.", cause=exc) from exc

    def _numeric_parse_ratio(self, series: pd.Series, config: SchemaDetectionConfig) -> float:
        try:
            sampled = self._sample_series(series, config.analysis_sample_size)
            normalized = sampled.map(self._safe_to_string).replace("", pd.NA).dropna()
            if normalized.empty:
                return 0.0

            cleaned = normalized.str.replace(",", "", regex=False)
            parsed = pd.to_numeric(cleaned, errors="coerce")
            success_count = int(parsed.notna().sum())
            return self._safe_ratio(success_count, int(normalized.shape[0]))
        except Exception as exc:
            self._log(logging.ERROR, f"Error | numeric parse ratio computation failed: {exc}")
            raise SchemaDetectionStageError("Failed to compute numeric parse ratio.", cause=exc) from exc

    def _datetime_parse_ratio(self, series: pd.Series, config: SchemaDetectionConfig) -> float:
        try:
            sampled = self._sample_series(series, config.analysis_sample_size)
            normalized = sampled.map(self._safe_to_string).replace("", pd.NA).dropna()
            if normalized.empty:
                return 0.0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                parsed = pd.to_datetime(normalized, errors="coerce")
            success_count = int(parsed.notna().sum())
            return self._safe_ratio(success_count, int(normalized.shape[0]))
        except Exception as exc:
            self._log(logging.ERROR, f"Error | datetime parse ratio computation failed: {exc}")
            raise SchemaDetectionStageError("Failed to compute datetime parse ratio.", cause=exc) from exc

    def _text_score(self, base_metadata: Dict[str, Any], config: SchemaDetectionConfig) -> float:
        try:
            score = 0.0

            if float(base_metadata.get("avg_string_length", 0.0)) >= config.text_avg_length_threshold:
                score += 0.35
            if float(base_metadata.get("avg_token_count", 0.0)) >= config.text_avg_token_count_threshold:
                score += 0.25
            if float(base_metadata.get("long_value_ratio", 0.0)) >= config.text_long_value_ratio_threshold:
                score += 0.20
            if float(base_metadata.get("unique_ratio", 0.0)) >= config.text_unique_ratio_threshold:
                score += 0.20

            return self._bounded_confidence(score)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | text score computation failed: {exc}")
            raise SchemaDetectionStageError("Failed to compute text inference score.", cause=exc) from exc

    def _categorical_confidence(
        self,
        base_metadata: Dict[str, Any],
        config: SchemaDetectionConfig,
    ) -> float:
        try:
            score = 0.30

            if float(base_metadata.get("unique_ratio", 0.0)) <= config.categorical_unique_ratio_threshold:
                score += 0.35
            if int(base_metadata.get("unique_count", 0)) <= config.categorical_unique_count_threshold:
                score += 0.20
            if float(base_metadata.get("avg_string_length", 0.0)) < config.text_avg_length_threshold:
                score += 0.10
            if float(base_metadata.get("avg_token_count", 0.0)) < config.text_avg_token_count_threshold:
                score += 0.05

            return self._bounded_confidence(score)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | categorical confidence computation failed: {exc}")
            raise SchemaDetectionStageError("Failed to compute categorical inference confidence.", cause=exc) from exc

    def _sample_values(self, series: pd.Series, count: int) -> List[Any]:
        try:
            if count <= 0 or series.empty:
                return []

            values: List[Any] = []
            for value in series.head(count):
                values.append(self._serialize_value(value))
            return values
        except Exception as exc:
            self._log(logging.ERROR, f"Error | sample value extraction failed: {exc}")
            raise SchemaDetectionStageError("Failed to extract sample values.", cause=exc) from exc

    def _serialize_value(self, value: Any) -> Any:
        try:
            if pd.isna(value):
                return None
            if hasattr(value, "isoformat"):
                return value.isoformat()
            if hasattr(value, "item"):
                return value.item()
            return value
        except Exception:
            return str(value)

    def _safe_to_string(self, value: Any) -> str:
        try:
            if pd.isna(value):
                return ""
            return str(value).strip()
        except Exception:
            return str(value)

    def _summarize_types(self, schema_columns: Sequence[SchemaColumn]) -> Dict[str, int]:
        try:
            counter = Counter(column.inferred_type for column in schema_columns)
            return {schema_type: int(counter.get(schema_type, 0)) for schema_type in sorted(self._supported_types)}
        except Exception as exc:
            self._log(logging.ERROR, f"Error | type distribution computation failed: {exc}")
            raise SchemaDetectionStageError("Failed to summarize inferred types.", cause=exc) from exc

    def _format_distribution(self, distribution: Dict[str, int]) -> str:
        try:
            ordered_items = [f"{schema_type}={count}" for schema_type, count in distribution.items()]
            return ", ".join(ordered_items)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | distribution formatting failed: {exc}")
            raise SchemaDetectionStageError("Failed to format schema type distribution.", cause=exc) from exc

    def _build_schema_metadata(
        self,
        dataframe: pd.DataFrame,
        schema_columns: Sequence[SchemaColumn],
        type_distribution: Dict[str, int],
        config: SchemaDetectionConfig,
        upstream_metadata: Optional[Dict[str, Any]],
        warnings: Sequence[str],
    ) -> Dict[str, Any]:
        try:
            metadata: Dict[str, Any] = {
                "stage": self.stage_name,
                "row_count": int(dataframe.shape[0]),
                "column_count": int(dataframe.shape[1]),
                "warnings": list(warnings),
            }

            if config.include_distribution_summary:
                metadata["type_distribution"] = dict(type_distribution)

            metadata["columns"] = [column.name for column in schema_columns]

            if upstream_metadata is not None:
                metadata["upstream_metadata"] = upstream_metadata

            return metadata
        except Exception as exc:
            self._log(logging.ERROR, f"Error | schema metadata construction failed: {exc}")
            raise SchemaDetectionStageError("Failed to build schema metadata.", cause=exc) from exc

    def _safe_ratio(self, numerator: int, denominator: int) -> float:
        try:
            if denominator <= 0:
                return 0.0
            return float(numerator) / float(denominator)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | ratio computation failed: {exc}")
            raise SchemaDetectionStageError("Failed to compute ratio.", cause=exc) from exc

    def _bounded_confidence(self, value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception as exc:
            self._log(logging.ERROR, f"Error | confidence normalization failed: {exc}")
            raise SchemaDetectionStageError("Failed to normalize inference confidence.", cause=exc) from exc

    def _log(self, level: int, message: str) -> None:
        try:
            self.logger.log(level, f"[{self.stage_name}] - {message}")
        except Exception:
            logging.getLogger(__name__).log(level, f"[{self.stage_name}] - {message}")


def detect_schema(
    dataframe: pd.DataFrame,
    config: Optional[SchemaDetectionConfig] = None,
    logger: Optional[logging.Logger] = None,
    upstream_metadata: Optional[Dict[str, Any]] = None,
) -> DatasetSchema:
    try:
        module = SchemaDetectionModule(config=config, logger=logger)
        return module.run(dataframe=dataframe, upstream_metadata=upstream_metadata)
    except SchemaDetectionStageError:
        raise
    except Exception as exc:
        raise SchemaDetectionStageError("Unhandled schema detection error.", cause=exc) from exc
