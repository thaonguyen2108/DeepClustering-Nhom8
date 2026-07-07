from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from .schema_detection import DatasetSchema, SchemaColumn


@dataclass
class PreprocessingConfig:
    numeric_scaler: str = "standard"
    categorical_encoder: str = "onehot"
    handle_unknown: str = "ignore"
    output_dtype: str = "float32"
    column_scaler_overrides: Dict[str, str] = field(default_factory=dict)
    column_encoder_overrides: Dict[str, str] = field(default_factory=dict)
    label_unknown_value: float = -1.0
    max_onehot_categories: Optional[int] = 100
    high_cardinality_fallback_encoder: str = "label"
    include_text_columns: bool = True
    keep_text_dataframe: bool = True
    infer_missing_schema_columns: bool = True
    text_avg_length_threshold: float = 25.0
    text_avg_token_count_threshold: float = 3.0
    text_unique_ratio_threshold: float = 0.30


@dataclass
class PreprocessingReport:
    stage: str
    row_count: int
    numeric_input_columns: List[str] = field(default_factory=list)
    categorical_input_columns: List[str] = field(default_factory=list)
    text_input_columns: List[str] = field(default_factory=list)
    skipped_columns: List[str] = field(default_factory=list)
    numeric_scaler_used: str = "standard"
    categorical_encoder_used: str = "onehot"
    numeric_output_shape: Tuple[int, int] = (0, 0)
    text_output_shape: Tuple[int, int] = (0, 0)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PreprocessingResult:
    X_numeric: np.ndarray
    X_text: pd.DataFrame
    feature_map: Dict[str, Any]
    preprocessing_report: PreprocessingReport
    feature_metadata: Dict[str, Any] = field(default_factory=dict)


class PreprocessingStageError(RuntimeError):
    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        try:
            super().__init__(message)
            self.cause = cause
        except Exception as exc:
            raise RuntimeError("Failed to initialize PreprocessingStageError.") from exc


class PreprocessingModule:
    stage_name = "PREPROCESSING"
    _supported_types = {"numeric", "categorical", "text", "unknown"}
    _scaler_aliases = {
        "standard": "standard",
        "standardscaler": "standard",
        "minmax": "minmax",
        "minmaxscaler": "minmax",
        "robust": "robust",
        "robustscaler": "robust",
    }
    _encoder_aliases = {
        "onehot": "onehot",
        "one_hot": "onehot",
        "one-hot": "onehot",
        "label": "label",
        "ordinal": "label",
        "labelencoder": "label",
        "ordinalencoder": "label",
    }

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        try:
            self.config = config or PreprocessingConfig()
            self.logger = logger or logging.getLogger(__name__)
        except Exception as exc:
            raise PreprocessingStageError("Failed to initialize preprocessing module.", cause=exc) from exc

    def run(
        self,
        dataframe: pd.DataFrame,
        schema: Any,
        config: Optional[PreprocessingConfig] = None,
    ) -> PreprocessingResult:
        try:
            active_config = config or self.config
            validated_frame, schema_lookup, warnings = self._validate_inputs(dataframe, schema, active_config)
            self._log(
                logging.INFO,
                f"Stage start | rows={validated_frame.shape[0]} | columns={validated_frame.shape[1]}",
            )

            column_groups = self._classify_columns(validated_frame, schema_lookup, warnings)
            self._log(
                logging.INFO,
                f"Column groups | numeric={len(column_groups['numeric'])} | "
                f"categorical={len(column_groups['categorical'])} | text={len(column_groups['text'])}",
            )

            numeric_blocks, numeric_feature_names, numeric_map, skipped_numeric = self._process_numeric_columns(
                validated_frame,
                column_groups["numeric"],
                active_config,
                warnings,
            )
            categorical_blocks, categorical_feature_names, categorical_map, skipped_categorical = (
                self._process_categorical_columns(
                    validated_frame,
                    column_groups["categorical"],
                    active_config,
                    warnings,
                )
            )
            skipped_columns = skipped_numeric + skipped_categorical + list(column_groups["skipped"])
            text_payload = self._build_text_payload(validated_frame, column_groups["text"], active_config)

            numeric_matrix_parts = numeric_blocks + categorical_blocks
            if numeric_matrix_parts:
                X_numeric = np.concatenate(numeric_matrix_parts, axis=1)
                X_numeric = X_numeric.astype(active_config.output_dtype, copy=False)
            else:
                X_numeric = np.empty((validated_frame.shape[0], 0), dtype=active_config.output_dtype)
                warnings.append("No numeric-compatible features were produced during preprocessing.")

            output_feature_names = numeric_feature_names + categorical_feature_names
            feature_map = self._build_feature_map(
                numeric_map=numeric_map,
                categorical_map=categorical_map,
                text_columns=column_groups["text"],
                output_feature_names=output_feature_names,
                skipped_columns=skipped_columns,
            )

            report = PreprocessingReport(
                stage=self.stage_name,
                row_count=int(validated_frame.shape[0]),
                numeric_input_columns=list(column_groups["numeric"]),
                categorical_input_columns=list(column_groups["categorical"]),
                text_input_columns=list(column_groups["text"]),
                skipped_columns=skipped_columns,
                numeric_scaler_used=active_config.numeric_scaler,
                categorical_encoder_used=active_config.categorical_encoder,
                numeric_output_shape=(int(X_numeric.shape[0]), int(X_numeric.shape[1])),
                text_output_shape=(int(text_payload.shape[0]), int(text_payload.shape[1])),
                warnings=warnings,
            )
            feature_metadata = self._build_feature_metadata(validated_frame, report, output_feature_names)

            self._log(
                logging.INFO,
                f"Methods used | numeric_scaler={active_config.numeric_scaler} | "
                f"categorical_encoder={active_config.categorical_encoder}",
            )
            if skipped_columns:
                self._log(logging.WARNING, f"Skipped columns | count={len(skipped_columns)} | columns={skipped_columns}")
            self._log(
                logging.INFO,
                f"Final feature shape | X_numeric={X_numeric.shape} | X_text={text_payload.shape}",
            )

            return PreprocessingResult(
                X_numeric=X_numeric,
                X_text=text_payload,
                feature_map=feature_map,
                preprocessing_report=report,
                feature_metadata=feature_metadata,
            )
        except PreprocessingStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage failed with {type(exc).__name__}: {exc}")
            raise PreprocessingStageError("Preprocessing failed.", cause=exc) from exc

    def _validate_inputs(
        self,
        dataframe: pd.DataFrame,
        schema: Any,
        config: PreprocessingConfig,
    ) -> Tuple[pd.DataFrame, Dict[str, SchemaColumn], List[str]]:
        try:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(f"Expected pandas DataFrame, received {type(dataframe).__name__}.")
            if dataframe.shape[1] == 0:
                raise ValueError("Input DataFrame does not contain any columns.")

            warnings: List[str] = []
            schema_columns = self._extract_schema_columns(schema, warnings)
            schema_lookup = {column.name: column for column in schema_columns}

            if config.infer_missing_schema_columns:
                for column_name in dataframe.columns:
                    column_key = str(column_name)
                    if column_key not in schema_lookup:
                        inferred_column = self._infer_missing_schema_column(
                            dataframe[column_name],
                            column_key,
                            config,
                        )
                        schema_lookup[column_key] = inferred_column
                        warnings.append(
                            f"Column '{column_key}' was not present in schema; preprocessing inferred "
                            f"type '{inferred_column.inferred_type}'."
                        )

            return dataframe, schema_lookup, warnings
        except Exception as exc:
            self._log(logging.ERROR, f"Error | input validation failed: {exc}")
            raise PreprocessingStageError("Invalid input for preprocessing.", cause=exc) from exc

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
            raise PreprocessingStageError("Failed to extract schema columns.", cause=exc) from exc

    def _infer_missing_schema_column(
        self,
        series: pd.Series,
        column_name: str,
        config: PreprocessingConfig,
    ) -> SchemaColumn:
        try:
            if is_numeric_dtype(series.dtype):
                inferred_type = "numeric"
            elif is_bool_dtype(series.dtype) or is_categorical_dtype(series.dtype) or is_datetime64_any_dtype(series.dtype):
                inferred_type = "categorical"
            else:
                non_null = series.dropna()
                if non_null.empty:
                    inferred_type = "unknown"
                else:
                    as_text = non_null.astype(str).str.strip()
                    avg_length = float(as_text.str.len().mean())
                    avg_token_count = float(as_text.map(lambda value: len(value.split())).mean())
                    unique_ratio = self._safe_ratio(int(as_text.nunique(dropna=True)), int(as_text.shape[0]))
                    if (
                        avg_length >= config.text_avg_length_threshold
                        and avg_token_count >= config.text_avg_token_count_threshold
                        and unique_ratio >= config.text_unique_ratio_threshold
                    ):
                        inferred_type = "text"
                    else:
                        inferred_type = "categorical"

            return SchemaColumn(
                name=column_name,
                inferred_type=inferred_type,
                metadata={"inference_source": "preprocessing_fallback"},
            )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | fallback schema inference failed for column={column_name}: {exc}")
            raise PreprocessingStageError("Failed to infer missing schema column.", cause=exc) from exc

    def _classify_columns(
        self,
        dataframe: pd.DataFrame,
        schema_lookup: Dict[str, SchemaColumn],
        warnings: List[str],
    ) -> Dict[str, List[str]]:
        try:
            column_groups = {"numeric": [], "categorical": [], "text": [], "skipped": []}

            for column_name in dataframe.columns:
                column_key = str(column_name)
                schema_column = schema_lookup.get(column_key)
                if schema_column is None:
                    warnings.append(f"Column '{column_key}' has no schema information and was skipped.")
                    column_groups["skipped"].append(column_key)
                    continue

                inferred_type = self._normalize_inferred_type(
                    schema_column.inferred_type,
                    column_key,
                    warnings,
                )

                if inferred_type == "numeric":
                    column_groups["numeric"].append(column_key)
                elif inferred_type == "categorical":
                    column_groups["categorical"].append(column_key)
                elif inferred_type == "text":
                    column_groups["text"].append(column_key)
                else:
                    warnings.append(f"Column '{column_key}' has unsupported schema type and was skipped.")
                    column_groups["skipped"].append(column_key)

            return column_groups
        except Exception as exc:
            self._log(logging.ERROR, f"Error | column classification failed: {exc}")
            raise PreprocessingStageError("Failed to classify preprocessing columns.", cause=exc) from exc

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
                    f"Column '{column_name}' has unsupported inferred type '{inferred_type}' and will be skipped."
                )
                return "unknown"
            return normalized
        except Exception as exc:
            self._log(logging.ERROR, f"Error | inferred type normalization failed for column={column_name}: {exc}")
            raise PreprocessingStageError("Failed to normalize inferred preprocessing type.", cause=exc) from exc

    def _process_numeric_columns(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: Sequence[str],
        config: PreprocessingConfig,
        warnings: List[str],
    ) -> Tuple[List[np.ndarray], List[str], Dict[str, Any], List[str]]:
        try:
            feature_blocks: List[np.ndarray] = []
            feature_names: List[str] = []
            feature_map: Dict[str, Any] = {}
            skipped_columns: List[str] = []

            for column_name in numeric_columns:
                try:
                    scaler_name = self._resolve_numeric_scaler_name(column_name, config)
                    scaler = self._make_scaler(scaler_name)
                    numeric_series = pd.to_numeric(dataframe[column_name], errors="coerce")
                    if numeric_series.isna().any():
                        raise ValueError("Numeric column contains non-numeric or missing values after cleaning.")

                    transformed = scaler.fit_transform(numeric_series.to_numpy().reshape(-1, 1))
                    feature_blocks.append(transformed.astype(config.output_dtype, copy=False))
                    feature_names.append(column_name)
                    feature_map[column_name] = {
                        "source_type": "numeric",
                        "transformation": "scaled",
                        "processor": scaler_name,
                        "output_features": [column_name],
                        "processor_state": {
                            "scale_": getattr(scaler, "scale_", np.array([])).tolist(),
                            "center_": getattr(
                                scaler,
                                "center_",
                                getattr(scaler, "mean_", np.array([])),
                            ).tolist(),
                        },
                        "skipped": False,
                    }
                except Exception as exc:
                    warnings.append(f"Numeric column '{column_name}' was skipped during preprocessing: {exc}")
                    skipped_columns.append(column_name)
                    feature_map[column_name] = {
                        "source_type": "numeric",
                        "transformation": "skipped",
                        "processor": None,
                        "output_features": [],
                        "skipped": True,
                        "error": str(exc),
                    }
                    self._log(logging.WARNING, f"Skipped numeric column | column={column_name} | reason={exc}")

            return feature_blocks, feature_names, feature_map, skipped_columns
        except Exception as exc:
            self._log(logging.ERROR, f"Error | numeric preprocessing failed: {exc}")
            raise PreprocessingStageError("Failed to preprocess numeric columns.", cause=exc) from exc

    def _process_categorical_columns(
        self,
        dataframe: pd.DataFrame,
        categorical_columns: Sequence[str],
        config: PreprocessingConfig,
        warnings: List[str],
    ) -> Tuple[List[np.ndarray], List[str], Dict[str, Any], List[str]]:
        try:
            feature_blocks: List[np.ndarray] = []
            feature_names: List[str] = []
            feature_map: Dict[str, Any] = {}
            skipped_columns: List[str] = []

            for column_name in categorical_columns:
                try:
                    encoder_name = self._resolve_categorical_encoder_name(column_name, config)
                    encoder = self._make_encoder(encoder_name, config)
                    column_values = dataframe[[column_name]].copy().astype("string")
                    unique_count = int(column_values[column_name].nunique(dropna=True))
                    fallback_encoder: Optional[str] = None
                    if (
                        encoder_name == "onehot"
                        and config.max_onehot_categories is not None
                        and unique_count > config.max_onehot_categories
                    ):
                        fallback_encoder = self._canonicalize_encoder(config.high_cardinality_fallback_encoder)
                        warnings.append(
                            f"Categorical column '{column_name}' exceeded max_onehot_categories="
                            f"{config.max_onehot_categories}. Falling back to '{fallback_encoder}'."
                        )
                        self._log(
                            logging.WARNING,
                            f"Skipped onehot expansion | column={column_name} | unique_count={unique_count} | "
                            f"fallback_encoder={fallback_encoder}",
                        )
                        encoder_name = fallback_encoder
                        encoder = self._make_encoder(encoder_name, config)

                    transformed = encoder.fit_transform(column_values)
                    transformed = np.asarray(transformed, dtype=config.output_dtype)

                    if encoder_name == "onehot":
                        output_features = encoder.get_feature_names_out([column_name]).tolist()
                        categories = [self._serialize_value(value) for value in encoder.categories_[0].tolist()]
                    else:
                        output_features = [column_name]
                        categories = [self._serialize_value(value) for value in encoder.categories_[0].tolist()]

                    feature_blocks.append(transformed)
                    feature_names.extend(output_features)
                    feature_map[column_name] = {
                        "source_type": "categorical",
                        "transformation": "encoded",
                        "processor": encoder_name,
                        "output_features": output_features,
                        "categories": categories,
                        "fallback_encoder": fallback_encoder,
                        "skipped": False,
                    }
                except Exception as exc:
                    warnings.append(f"Categorical column '{column_name}' was skipped during preprocessing: {exc}")
                    skipped_columns.append(column_name)
                    feature_map[column_name] = {
                        "source_type": "categorical",
                        "transformation": "skipped",
                        "processor": None,
                        "output_features": [],
                        "skipped": True,
                        "error": str(exc),
                    }
                    self._log(logging.WARNING, f"Skipped categorical column | column={column_name} | reason={exc}")

            return feature_blocks, feature_names, feature_map, skipped_columns
        except Exception as exc:
            self._log(logging.ERROR, f"Error | categorical preprocessing failed: {exc}")
            raise PreprocessingStageError("Failed to preprocess categorical columns.", cause=exc) from exc

    def _build_text_payload(
        self,
        dataframe: pd.DataFrame,
        text_columns: Sequence[str],
        config: PreprocessingConfig,
    ) -> pd.DataFrame:
        try:
            if not config.include_text_columns or not config.keep_text_dataframe or not text_columns:
                return pd.DataFrame(index=dataframe.index)
            return dataframe.loc[:, list(text_columns)].copy()
        except Exception as exc:
            self._log(logging.ERROR, f"Error | text payload construction failed: {exc}")
            raise PreprocessingStageError("Failed to build text preprocessing payload.", cause=exc) from exc

    def _build_feature_map(
        self,
        numeric_map: Dict[str, Any],
        categorical_map: Dict[str, Any],
        text_columns: Sequence[str],
        output_feature_names: Sequence[str],
        skipped_columns: Sequence[str],
    ) -> Dict[str, Any]:
        try:
            column_transformations: Dict[str, Any] = {}
            column_transformations.update(numeric_map)
            column_transformations.update(categorical_map)
            for column_name in text_columns:
                column_transformations[column_name] = {
                    "source_type": "text",
                    "transformation": "passthrough",
                    "processor": None,
                    "output_features": [],
                    "skipped": False,
                }

            return {
                "column_transformations": column_transformations,
                "column_groups": {
                    "numeric": sorted([name for name, value in numeric_map.items() if not value["skipped"]]),
                    "categorical": sorted([name for name, value in categorical_map.items() if not value["skipped"]]),
                    "text": list(text_columns),
                    "skipped": list(skipped_columns),
                },
                "output_feature_names": list(output_feature_names),
            }
        except Exception as exc:
            self._log(logging.ERROR, f"Error | feature map construction failed: {exc}")
            raise PreprocessingStageError("Failed to build preprocessing feature map.", cause=exc) from exc

    def _build_feature_metadata(
        self,
        dataframe: pd.DataFrame,
        report: PreprocessingReport,
        output_feature_names: Sequence[str],
    ) -> Dict[str, Any]:
        try:
            return {
                "stage": self.stage_name,
                "row_count": int(dataframe.shape[0]),
                "input_column_count": int(dataframe.shape[1]),
                "numeric_input_columns": list(report.numeric_input_columns),
                "categorical_input_columns": list(report.categorical_input_columns),
                "text_input_columns": list(report.text_input_columns),
                "skipped_columns": list(report.skipped_columns),
                "output_feature_names": list(output_feature_names),
                "numeric_output_shape": list(report.numeric_output_shape),
                "text_output_shape": list(report.text_output_shape),
                "warnings": list(report.warnings),
            }
        except Exception as exc:
            self._log(logging.ERROR, f"Error | feature metadata construction failed: {exc}")
            raise PreprocessingStageError("Failed to build preprocessing metadata.", cause=exc) from exc

    def _resolve_numeric_scaler_name(self, column_name: str, config: PreprocessingConfig) -> str:
        try:
            return self._canonicalize_scaler(
                config.column_scaler_overrides.get(column_name, config.numeric_scaler)
            )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | numeric scaler resolution failed for column={column_name}: {exc}")
            raise PreprocessingStageError("Failed to resolve numeric scaler.", cause=exc) from exc

    def _resolve_categorical_encoder_name(self, column_name: str, config: PreprocessingConfig) -> str:
        try:
            return self._canonicalize_encoder(
                config.column_encoder_overrides.get(column_name, config.categorical_encoder)
            )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | categorical encoder resolution failed for column={column_name}: {exc}")
            raise PreprocessingStageError("Failed to resolve categorical encoder.", cause=exc) from exc

    def _canonicalize_scaler(self, scaler_name: str) -> str:
        try:
            normalized = str(scaler_name).strip().lower().replace("-", "")
            if normalized in self._scaler_aliases:
                return self._scaler_aliases[normalized]
            raise ValueError(f"Unsupported numeric scaler '{scaler_name}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | scaler canonicalization failed for scaler={scaler_name}: {exc}")
            raise PreprocessingStageError("Failed to canonicalize numeric scaler.", cause=exc) from exc

    def _canonicalize_encoder(self, encoder_name: str) -> str:
        try:
            normalized = str(encoder_name).strip().lower().replace("-", "_")
            if normalized in self._encoder_aliases:
                return self._encoder_aliases[normalized]
            raise ValueError(f"Unsupported categorical encoder '{encoder_name}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | encoder canonicalization failed for encoder={encoder_name}: {exc}")
            raise PreprocessingStageError("Failed to canonicalize categorical encoder.", cause=exc) from exc

    def _make_scaler(self, scaler_name: str):
        try:
            if scaler_name == "standard":
                return StandardScaler()
            if scaler_name == "minmax":
                return MinMaxScaler()
            if scaler_name == "robust":
                return RobustScaler()
            raise ValueError(f"Unsupported scaler '{scaler_name}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | scaler construction failed for scaler={scaler_name}: {exc}")
            raise PreprocessingStageError("Failed to construct numeric scaler.", cause=exc) from exc

    def _make_encoder(self, encoder_name: str, config: PreprocessingConfig):
        try:
            if encoder_name == "onehot":
                return OneHotEncoder(
                    handle_unknown=config.handle_unknown,
                    sparse_output=False,
                    dtype=np.dtype(config.output_dtype),
                )
            if encoder_name == "label":
                if config.handle_unknown == "ignore":
                    return OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=int(config.label_unknown_value),
                        encoded_missing_value=int(config.label_unknown_value),
                        dtype=np.int64,
                    )
                return OrdinalEncoder(
                    handle_unknown="error",
                    dtype=np.int64,
                )
            raise ValueError(f"Unsupported encoder '{encoder_name}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | encoder construction failed for encoder={encoder_name}: {exc}")
            raise PreprocessingStageError("Failed to construct categorical encoder.", cause=exc) from exc

    def _safe_ratio(self, numerator: int, denominator: int) -> float:
        try:
            if denominator <= 0:
                return 0.0
            return float(numerator) / float(denominator)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | ratio computation failed: {exc}")
            raise PreprocessingStageError("Failed to compute preprocessing ratio.", cause=exc) from exc

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

    def _log(self, level: int, message: str) -> None:
        try:
            self.logger.log(level, f"[{self.stage_name}] - {message}")
        except Exception:
            logging.getLogger(__name__).log(level, f"[{self.stage_name}] - {message}")


def preprocess_features(
    dataframe: pd.DataFrame,
    schema: Any,
    config: Optional[PreprocessingConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> PreprocessingResult:
    try:
        module = PreprocessingModule(config=config, logger=logger)
        return module.run(dataframe=dataframe, schema=schema)
    except PreprocessingStageError:
        raise
    except Exception as exc:
        raise PreprocessingStageError("Unhandled preprocessing error.", cause=exc) from exc
