from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from Embeddings.embeddings import EmbeddingEngine


@dataclass
class EmbeddingConfig:
    batch_size: int = 32
    merge_strategy: str = "concat_text"
    handle_empty: str = "replace"
    default_text: str = "unknown"
    text_separator: str = " "
    output_dtype: str = "float32"
    model_name: str = "intfloat/multilingual-e5-small"
    device: Optional[str] = None
    normalize: bool = True
    auto_batch: bool = True
    safety_margin: float = 0.8
    force_offline: bool = True
    metadata_row_index_sample_size: int = 10
    zero_tolerance: float = 1e-8
    min_variance_threshold: float = 1e-12
    variance_check_min_rows: int = 2


@dataclass
class EmbeddingReport:
    stage: str
    row_count: int
    text_columns: List[str] = field(default_factory=list)
    text_column_count: int = 0
    merge_strategy: str = "concat_text"
    batch_size: int = 32
    device: Optional[str] = None
    model_name: str = "intfloat/multilingual-e5-small"
    base_embedding_dimension: int = 0
    embedding_output_shape: Tuple[int, int] = (0, 0)
    skipped_columns: List[str] = field(default_factory=list)
    skipped_row_count: int = 0
    skipped_row_indices_sample: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EmbeddingResult:
    X_embedding: np.ndarray
    embedding_metadata: Dict[str, Any]
    embedding_report: EmbeddingReport


class EmbeddingStageError(RuntimeError):
    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        try:
            super().__init__(message)
            self.cause = cause
        except Exception as exc:
            raise RuntimeError("Failed to initialize EmbeddingStageError.") from exc


class EmbeddingModule:
    stage_name = "EMBEDDING"
    _merge_strategy_aliases = {
        "concat_text": "concat_text",
        "concattext": "concat_text",
        "text_concat": "concat_text",
        "concat_vector": "concat_vector",
        "concatvector": "concat_vector",
        "vector_concat": "concat_vector",
    }
    _handle_empty_aliases = {
        "replace": "replace",
        "fill": "replace",
        "default": "replace",
        "skip": "skip",
        "ignore": "skip",
    }

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        try:
            self.config = config or EmbeddingConfig()
            self.logger = logger or logging.getLogger(__name__)
        except Exception as exc:
            raise EmbeddingStageError("Failed to initialize embedding module.", cause=exc) from exc

    def run(
        self,
        X_text: Optional[pd.DataFrame],
        feature_map: Optional[Dict[str, Any]] = None,
        preprocessing_metadata: Optional[Dict[str, Any]] = None,
        config: Optional[EmbeddingConfig] = None,
    ) -> EmbeddingResult:
        try:
            active_config = config or self.config
            validated_config = self._validate_config(active_config)
            text_frame, text_columns, row_count, warnings = self._validate_inputs(
                X_text=X_text,
                feature_map=feature_map,
                preprocessing_metadata=preprocessing_metadata,
            )
            merge_strategy = self._canonicalize_merge_strategy(validated_config.merge_strategy)

            self._log(logging.INFO, f"Stage start | rows={row_count} | text_columns={len(text_columns)}")

            if not text_columns:
                self._log(logging.INFO, "No text features detected. Skipping embedding stage.")
                result = self._build_skipped_result(
                    row_count=row_count,
                    text_columns=[],
                    merge_strategy=merge_strategy,
                    config=validated_config,
                    warnings=warnings,
                )
                self._validate_stage_output(result)
                return result

            self._log(logging.INFO, f"Text columns detected | count={len(text_columns)} | columns={text_columns}")

            with self._make_engine(validated_config) as engine:
                self._log(
                    logging.INFO,
                    f"Batch processing | requested_batch_size={validated_config.batch_size} | "
                    f"device={engine.device} | auto_batch={validated_config.auto_batch}",
                )

                if merge_strategy == "concat_text":
                    X_embedding, skipped_columns, skipped_rows = self._embed_concat_text(
                        text_frame=text_frame,
                        text_columns=text_columns,
                        config=validated_config,
                        engine=engine,
                        warnings=warnings,
                    )
                else:
                    X_embedding, skipped_columns, skipped_rows = self._embed_concat_vector(
                        text_frame=text_frame,
                        text_columns=text_columns,
                        config=validated_config,
                        engine=engine,
                        warnings=warnings,
                    )

                X_embedding = self._ensure_numeric_matrix(
                    matrix=X_embedding,
                    row_count=row_count,
                    output_dtype=validated_config.output_dtype,
                )
                X_embedding = self._validate_embedding_matrix(
                    matrix=X_embedding,
                    row_count=row_count,
                    config=validated_config,
                    scope="stage_output",
                )
                output_dimension = int(X_embedding.shape[1]) if X_embedding.ndim == 2 else 0
                self._log(logging.INFO, f"Embedding output shape | shape={tuple(X_embedding.shape)}")

                if skipped_columns:
                    self._log(
                        logging.WARNING,
                        f"Skipped columns | count={len(skipped_columns)} | columns={sorted(skipped_columns)}",
                    )
                if skipped_rows:
                    self._log(
                        logging.WARNING,
                        f"Skipped rows | count={len(skipped_rows)} | "
                        f"sample={skipped_rows[:validated_config.metadata_row_index_sample_size]}",
                    )

                report = EmbeddingReport(
                    stage=self.stage_name,
                    row_count=row_count,
                    text_columns=list(text_columns),
                    text_column_count=int(len(text_columns)),
                    merge_strategy=merge_strategy,
                    batch_size=int(validated_config.batch_size),
                    device=str(engine.device),
                    model_name=str(validated_config.model_name),
                    base_embedding_dimension=int(getattr(engine, "embedding_dim", 0) or 0),
                    embedding_output_shape=(int(X_embedding.shape[0]), output_dimension),
                    skipped_columns=sorted(skipped_columns),
                    skipped_row_count=int(len(skipped_rows)),
                    skipped_row_indices_sample=skipped_rows[: validated_config.metadata_row_index_sample_size],
                    warnings=warnings,
                )
                metadata = self._build_embedding_metadata(report=report, config=validated_config)
                result = EmbeddingResult(
                    X_embedding=X_embedding,
                    embedding_metadata=metadata,
                    embedding_report=report,
                )
                self._validate_stage_output(result)
                return result
        except EmbeddingStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage failed with {type(exc).__name__}: {exc}")
            raise EmbeddingStageError("Embedding failed.", cause=exc) from exc

    def _validate_config(self, config: EmbeddingConfig) -> EmbeddingConfig:
        try:
            config.batch_size = max(1, int(config.batch_size))
            config.output_dtype = str(np.dtype(config.output_dtype))
            if float(config.safety_margin) <= 0:
                raise ValueError("safety_margin must be greater than 0.")
            if int(config.metadata_row_index_sample_size) < 0:
                raise ValueError("metadata_row_index_sample_size cannot be negative.")
            if not str(config.model_name).strip():
                raise ValueError("model_name cannot be empty.")
            if float(config.zero_tolerance) < 0:
                raise ValueError("zero_tolerance cannot be negative.")
            if float(config.min_variance_threshold) < 0:
                raise ValueError("min_variance_threshold cannot be negative.")
            if int(config.variance_check_min_rows) < 2:
                raise ValueError("variance_check_min_rows must be at least 2.")
            config.metadata_row_index_sample_size = int(config.metadata_row_index_sample_size)
            config.model_name = str(config.model_name).strip()
            config.zero_tolerance = float(config.zero_tolerance)
            config.min_variance_threshold = float(config.min_variance_threshold)
            config.variance_check_min_rows = int(config.variance_check_min_rows)
            config.merge_strategy = self._canonicalize_merge_strategy(config.merge_strategy)
            config.handle_empty = self._canonicalize_handle_empty(config.handle_empty)
            return config
        except Exception as exc:
            self._log(logging.ERROR, f"Error | invalid embedding config: {exc}")
            raise EmbeddingStageError("Invalid embedding configuration.", cause=exc) from exc

    def _validate_inputs(
        self,
        X_text: Optional[pd.DataFrame],
        feature_map: Optional[Dict[str, Any]],
        preprocessing_metadata: Optional[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, List[str], int, List[str]]:
        try:
            warnings: List[str] = []

            declared_text_columns = self._extract_declared_text_columns(feature_map, preprocessing_metadata, warnings)
            fallback_row_count = self._extract_row_count(preprocessing_metadata)

            if X_text is None:
                empty_frame = pd.DataFrame(index=pd.RangeIndex(fallback_row_count))
                if declared_text_columns:
                    warnings.append("X_text was None; embedding stage received declared text columns but no text payload.")
                return empty_frame, [], int(fallback_row_count), warnings

            if not isinstance(X_text, pd.DataFrame):
                raise TypeError(f"Expected X_text to be a pandas DataFrame, received {type(X_text).__name__}.")

            text_frame = X_text.copy()
            text_frame.columns = [str(column) for column in text_frame.columns]
            row_count = int(text_frame.shape[0])
            available_columns = list(text_frame.columns)

            if declared_text_columns:
                resolved_columns = [column for column in declared_text_columns if column in text_frame.columns]
                missing_columns = [column for column in declared_text_columns if column not in text_frame.columns]
                if missing_columns:
                    warnings.append(
                        f"Declared text columns are missing from X_text and will be skipped: {missing_columns}"
                    )
                undeclared_columns = [column for column in available_columns if column not in resolved_columns]
                if undeclared_columns:
                    warnings.append(
                        f"X_text contains columns not declared in preprocessing metadata; treating them as text: {undeclared_columns}"
                    )
                    resolved_columns.extend(undeclared_columns)
            else:
                resolved_columns = available_columns

            return text_frame, resolved_columns, row_count, warnings
        except Exception as exc:
            self._log(logging.ERROR, f"Error | input validation failed: {exc}")
            raise EmbeddingStageError("Invalid input for embedding module.", cause=exc) from exc

    def _extract_declared_text_columns(
        self,
        feature_map: Optional[Dict[str, Any]],
        preprocessing_metadata: Optional[Dict[str, Any]],
        warnings: List[str],
    ) -> List[str]:
        try:
            declared: List[str] = []
            if isinstance(feature_map, dict):
                groups = feature_map.get("column_groups", {})
                for column in groups.get("text", []):
                    declared.append(str(column))

                if not declared:
                    transformations = feature_map.get("column_transformations", {})
                    for column_name, transformation in transformations.items():
                        if str(transformation.get("source_type")) == "text":
                            declared.append(str(column_name))

            if isinstance(preprocessing_metadata, dict):
                for column in preprocessing_metadata.get("text_input_columns", []):
                    declared.append(str(column))

            deduplicated: List[str] = []
            seen = set()
            for column in declared:
                if column not in seen:
                    deduplicated.append(column)
                    seen.add(column)

            return deduplicated
        except Exception as exc:
            warnings.append(f"Failed to extract declared text columns from preprocessing metadata: {exc}")
            self._log(logging.WARNING, f"Fallback usage | declared text column extraction failed: {exc}")
            return []

    def _extract_row_count(self, preprocessing_metadata: Optional[Dict[str, Any]]) -> int:
        try:
            if isinstance(preprocessing_metadata, dict):
                value = preprocessing_metadata.get("row_count")
                if value is not None:
                    return max(0, int(value))
            return 0
        except Exception:
            return 0

    def _embed_concat_text(
        self,
        text_frame: pd.DataFrame,
        text_columns: Sequence[str],
        config: EmbeddingConfig,
        engine: EmbeddingEngine,
        warnings: List[str],
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        try:
            skipped_columns: List[str] = []
            usable_columns: List[str] = []

            for column_name in text_columns:
                if column_name not in text_frame.columns:
                    skipped_columns.append(str(column_name))
                    warnings.append(f"Text column '{column_name}' is missing from X_text and was skipped.")
                    self._log(logging.WARNING, f"Skipped column | column={column_name} | reason=missing_from_X_text")
                    continue
                usable_columns.append(str(column_name))

            if not usable_columns:
                return self._build_empty_matrix(int(text_frame.shape[0]), config.output_dtype), skipped_columns, []

            prepared_texts: List[Optional[str]] = []
            skipped_rows: List[int] = []

            for row_position in range(int(text_frame.shape[0])):
                try:
                    merged_value = self._merge_row_text(
                        row=text_frame.iloc[row_position],
                        text_columns=usable_columns,
                        config=config,
                    )
                    if merged_value is None:
                        skipped_rows.append(row_position)
                    prepared_texts.append(merged_value)
                except Exception as exc:
                    skipped_rows.append(row_position)
                    warnings.append(f"Row {row_position} failed text merge and was skipped: {exc}")
                    self._log(
                        logging.WARNING,
                        f"Skipped row | scope=concat_text | row_index={row_position} | reason={exc}",
                    )
                    prepared_texts.append(None)

            embedded_matrix, failed_rows = self._embed_optional_texts(
                texts=prepared_texts,
                engine=engine,
                config=config,
                scope="concat_text",
                warnings=warnings,
            )
            return embedded_matrix, skipped_columns, self._merge_index_lists(skipped_rows, failed_rows)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | concat_text embedding failed: {exc}")
            raise EmbeddingStageError("Failed to embed concatenated text columns.", cause=exc) from exc

    def _embed_concat_vector(
        self,
        text_frame: pd.DataFrame,
        text_columns: Sequence[str],
        config: EmbeddingConfig,
        engine: EmbeddingEngine,
        warnings: List[str],
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        try:
            row_count = int(text_frame.shape[0])
            embedded_blocks: List[np.ndarray] = []
            skipped_columns: List[str] = []
            skipped_rows: List[int] = []

            for column_name in text_columns:
                if column_name not in text_frame.columns:
                    skipped_columns.append(str(column_name))
                    warnings.append(f"Text column '{column_name}' is missing from X_text and was skipped.")
                    self._log(logging.WARNING, f"Skipped column | column={column_name} | reason=missing_from_X_text")
                    continue

                try:
                    prepared_texts = self._prepare_single_column_texts(
                        series=text_frame[column_name],
                        config=config,
                    )
                    column_matrix, column_failed_rows = self._embed_optional_texts(
                        texts=prepared_texts,
                        engine=engine,
                        config=config,
                        scope=f"column={column_name}",
                        warnings=warnings,
                    )
                    if column_matrix.shape[0] != row_count:
                        raise ValueError(
                            f"Embedded column '{column_name}' returned {column_matrix.shape[0]} rows, expected {row_count}."
                        )
                    embedded_blocks.append(column_matrix)
                    skipped_rows = self._merge_index_lists(
                        skipped_rows,
                        [index for index, text in enumerate(prepared_texts) if text is None],
                        column_failed_rows,
                    )
                except Exception as exc:
                    skipped_columns.append(str(column_name))
                    warnings.append(f"Text column '{column_name}' failed embedding and was skipped: {exc}")
                    self._log(logging.WARNING, f"Skipped column | column={column_name} | reason={exc}")

            if not embedded_blocks:
                return self._build_empty_matrix(row_count, config.output_dtype), skipped_columns, skipped_rows

            combined = np.concatenate(embedded_blocks, axis=1)
            return combined, skipped_columns, skipped_rows
        except Exception as exc:
            self._log(logging.ERROR, f"Error | concat_vector embedding failed: {exc}")
            raise EmbeddingStageError("Failed to embed text columns as concatenated vectors.", cause=exc) from exc

    def _merge_row_text(
        self,
        row: pd.Series,
        text_columns: Sequence[str],
        config: EmbeddingConfig,
    ) -> Optional[str]:
        try:
            parts: List[str] = []
            for column_name in text_columns:
                normalized = self._normalize_text_value(row.get(column_name))
                if normalized:
                    parts.append(normalized)

            if parts:
                return config.text_separator.join(parts).strip()

            if config.handle_empty == "replace":
                return str(config.default_text)

            return None
        except Exception as exc:
            self._log(logging.ERROR, f"Error | row text merge failed: {exc}")
            raise EmbeddingStageError("Failed to merge text row.", cause=exc) from exc

    def _prepare_single_column_texts(
        self,
        series: pd.Series,
        config: EmbeddingConfig,
    ) -> List[Optional[str]]:
        try:
            prepared: List[Optional[str]] = []
            for value in series:
                normalized = self._normalize_text_value(value)
                if normalized:
                    prepared.append(normalized)
                    continue
                if config.handle_empty == "replace":
                    prepared.append(str(config.default_text))
                else:
                    prepared.append(None)
            return prepared
        except Exception as exc:
            self._log(logging.ERROR, f"Error | column text preparation failed: {exc}")
            raise EmbeddingStageError("Failed to prepare text column for embedding.", cause=exc) from exc

    def _embed_optional_texts(
        self,
        texts: Sequence[Optional[str]],
        engine: EmbeddingEngine,
        config: EmbeddingConfig,
        scope: str,
        warnings: List[str],
    ) -> Tuple[np.ndarray, List[int]]:
        try:
            row_count = int(len(texts))
            embedding_dim = int(getattr(engine, "embedding_dim", 0) or 0)
            output = np.zeros((row_count, embedding_dim), dtype=np.dtype(config.output_dtype))

            valid_positions = [index for index, text in enumerate(texts) if text is not None]
            skipped_rows = [index for index, text in enumerate(texts) if text is None]

            if not valid_positions:
                self._log(logging.WARNING, f"Skipped case | scope={scope} | all rows empty or skipped")
                return output, skipped_rows

            payload = [str(texts[index]) for index in valid_positions]
            embedded_payload, failed_relative_rows = self._embed_text_batch(
                texts=payload,
                engine=engine,
                config=config,
                scope=scope,
                warnings=warnings,
            )
            output[valid_positions, :] = embedded_payload

            failed_rows = [valid_positions[index] for index in failed_relative_rows]
            return output, self._merge_index_lists(skipped_rows, failed_rows)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | optional text embedding failed for scope={scope}: {exc}")
            raise EmbeddingStageError("Failed to embed optional text payload.", cause=exc) from exc

    def _embed_text_batch(
        self,
        texts: Sequence[str],
        engine: EmbeddingEngine,
        config: EmbeddingConfig,
        scope: str,
        warnings: List[str],
    ) -> Tuple[np.ndarray, List[int]]:
        try:
            if not texts:
                return self._build_empty_matrix(0, config.output_dtype, int(engine.embedding_dim)), []

            chunks = [{"text": text, "row_index": index} for index, text in enumerate(texts)]
            self._log(
                logging.INFO,
                f"Batch processing | scope={scope} | samples={len(texts)} | batch_size={config.batch_size}",
            )

            try:
                embedded_chunks = engine.embed_chunks(chunks=chunks, batch_size=config.batch_size)
                vectors = [np.asarray(chunk["vector"], dtype=np.dtype(config.output_dtype)) for chunk in embedded_chunks]
                matrix = (
                    np.vstack(vectors)
                    if vectors
                    else self._build_empty_matrix(0, config.output_dtype, int(engine.embedding_dim))
                )
                matrix = self._ensure_numeric_matrix(
                    matrix=matrix,
                    row_count=int(len(texts)),
                    output_dtype=config.output_dtype,
                )
            except Exception as exc:
                warnings.append(f"Batch embedding failed for '{scope}'. Falling back to row-level embedding: {exc}")
                self._log(
                    logging.WARNING,
                    f"Fallback usage | scope={scope} | mode=row_level | reason={type(exc).__name__}: {exc}",
                )
                return self._embed_rows_individually(
                    texts=texts,
                    engine=engine,
                    config=config,
                    scope=scope,
                    warnings=warnings,
                )
            matrix = self._validate_embedding_matrix(
                matrix=matrix,
                row_count=int(len(texts)),
                config=config,
                scope=scope,
            )
            return matrix, []
        except Exception as exc:
            self._log(logging.ERROR, f"Error | batch embedding failed for scope={scope}: {exc}")
            raise EmbeddingStageError("Failed to embed text batch.", cause=exc) from exc

    def _embed_rows_individually(
        self,
        texts: Sequence[str],
        engine: EmbeddingEngine,
        config: EmbeddingConfig,
        scope: str,
        warnings: List[str],
    ) -> Tuple[np.ndarray, List[int]]:
        try:
            embedding_dim = int(getattr(engine, "embedding_dim", 0) or 0)
            output = np.zeros((len(texts), embedding_dim), dtype=np.dtype(config.output_dtype))
            failed_rows: List[int] = []

            for row_index, text in enumerate(texts):
                try:
                    embedded_row = engine.embed_chunks(
                        chunks=[{"text": str(text), "row_index": row_index}],
                        batch_size=1,
                    )
                    vector = np.asarray(embedded_row[0]["vector"], dtype=np.dtype(config.output_dtype)).reshape(1, -1)
                    vector = self._ensure_numeric_matrix(
                        matrix=vector,
                        row_count=1,
                        output_dtype=config.output_dtype,
                    )
                    output[row_index, :] = vector[0]
                except Exception as exc:
                    failed_rows.append(int(row_index))
                    warnings.append(
                        f"Row {row_index} failed embedding in '{scope}' and was replaced with zeros: {exc}"
                    )
                    self._log(
                        logging.WARNING,
                        f"Skipped row | scope={scope} | row_index={row_index} | fallback=zeros | reason={exc}",
                    )

            output = self._validate_embedding_matrix(
                matrix=output,
                row_count=int(len(texts)),
                config=config,
                scope=f"{scope}:row_level",
            )
            return output, failed_rows
        except Exception as exc:
            self._log(logging.ERROR, f"Error | row-level embedding fallback failed for scope={scope}: {exc}")
            raise EmbeddingStageError("Failed to perform row-level embedding fallback.", cause=exc) from exc

    def _ensure_numeric_matrix(
        self,
        matrix: np.ndarray,
        row_count: int,
        output_dtype: str,
    ) -> np.ndarray:
        try:
            array = np.asarray(matrix, dtype=np.dtype(output_dtype))
            if array.ndim == 1:
                if row_count == 1:
                    array = array.reshape(1, -1)
                elif row_count == 0 and array.size == 0:
                    array = array.reshape(0, 0)
                else:
                    raise ValueError("Embedding output must be a 2D numeric matrix.")

            if array.ndim != 2:
                raise ValueError("Embedding output must be a 2D numeric matrix.")
            if int(array.shape[0]) != int(row_count):
                raise ValueError(f"Embedding row count mismatch: expected {row_count}, received {array.shape[0]}.")
            return array
        except Exception as exc:
            self._log(logging.ERROR, f"Error | numeric matrix validation failed: {exc}")
            raise EmbeddingStageError("Failed to validate embedding output matrix.", cause=exc) from exc

    def _build_skipped_result(
        self,
        row_count: int,
        text_columns: Sequence[str],
        merge_strategy: str,
        config: EmbeddingConfig,
        warnings: List[str],
    ) -> EmbeddingResult:
        try:
            empty_matrix = self._build_empty_matrix(row_count, config.output_dtype)
            report = EmbeddingReport(
                stage=self.stage_name,
                row_count=int(row_count),
                text_columns=list(text_columns),
                text_column_count=int(len(text_columns)),
                merge_strategy=merge_strategy,
                batch_size=int(config.batch_size),
                device=None,
                model_name=str(config.model_name),
                base_embedding_dimension=0,
                embedding_output_shape=(int(empty_matrix.shape[0]), int(empty_matrix.shape[1])),
                skipped_columns=[],
                skipped_row_count=0,
                skipped_row_indices_sample=[],
                warnings=warnings,
            )
            metadata = self._build_embedding_metadata(report=report, config=config)
            return EmbeddingResult(
                X_embedding=empty_matrix,
                embedding_metadata=metadata,
                embedding_report=report,
            )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | skipped result construction failed: {exc}")
            raise EmbeddingStageError("Failed to build skipped embedding result.", cause=exc) from exc

    def _build_embedding_metadata(
        self,
        report: EmbeddingReport,
        config: EmbeddingConfig,
    ) -> Dict[str, Any]:
        try:
            return {
                "stage": self.stage_name,
                "row_count": int(report.row_count),
                "text_columns": list(report.text_columns),
                "text_column_count": int(report.text_column_count),
                "merge_strategy": str(report.merge_strategy),
                "batch_size": int(report.batch_size),
                "model_name": str(report.model_name),
                "device": report.device,
                "base_embedding_dimension": int(report.base_embedding_dimension),
                "embedding_dimension": int(report.embedding_output_shape[1]),
                "output_shape": list(report.embedding_output_shape),
                "handle_empty": str(config.handle_empty),
                "default_text": str(config.default_text),
                "skipped_columns": list(report.skipped_columns),
                "skipped_row_count": int(report.skipped_row_count),
                "skipped_row_indices_sample": list(report.skipped_row_indices_sample),
                "warnings": list(report.warnings),
            }
        except Exception as exc:
            self._log(logging.ERROR, f"Error | embedding metadata construction failed: {exc}")
            raise EmbeddingStageError("Failed to build embedding metadata.", cause=exc) from exc

    def _make_engine(self, config: EmbeddingConfig) -> EmbeddingEngine:
        try:
            self._ensure_embedding_dependencies()
            from Embeddings.embeddings import EmbeddingEngine

            engine = EmbeddingEngine(
                model_name=config.model_name,
                device=config.device,
                batch_size=config.batch_size,
                normalize=config.normalize,
                auto_batch=config.auto_batch,
                safety_margin=config.safety_margin,
                force_offline=config.force_offline,
            )
            self._validate_engine(engine)
            return engine
        except EmbeddingStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | embedding engine initialization failed: {exc}")
            raise EmbeddingStageError("Failed to initialize embedding engine.", cause=exc) from exc

    def _build_empty_matrix(
        self,
        row_count: int,
        output_dtype: str,
        column_count: int = 0,
    ) -> np.ndarray:
        try:
            return np.empty((int(row_count), int(column_count)), dtype=np.dtype(output_dtype))
        except Exception as exc:
            self._log(logging.ERROR, f"Error | empty matrix construction failed: {exc}")
            raise EmbeddingStageError("Failed to construct empty embedding matrix.", cause=exc) from exc

    def _merge_index_lists(self, *index_lists: Sequence[int]) -> List[int]:
        try:
            merged = set()
            for index_list in index_lists:
                merged.update(int(index) for index in index_list)
            return sorted(merged)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | skipped index merge failed: {exc}")
            raise EmbeddingStageError("Failed to merge skipped row indices.", cause=exc) from exc

    def _ensure_embedding_dependencies(self) -> None:
        try:
            required_dependencies = {
                "torch": "torch",
                "transformers": "transformers",
                "sentence_transformers": "sentence-transformers",
            }
            missing_packages = [
                install_name
                for module_name, install_name in required_dependencies.items()
                if importlib.util.find_spec(module_name) is None
            ]
            if missing_packages:
                joined_packages = " ".join(missing_packages)
                raise EmbeddingStageError(
                    "Embedding stage requires installed dependencies: "
                    f"{', '.join(missing_packages)}. Install with `pip install {joined_packages}`."
                )
        except EmbeddingStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | embedding dependencies unavailable: {exc}")
            raise EmbeddingStageError("Embedding runtime dependencies are unavailable.", cause=exc) from exc

    def _validate_engine(self, engine: Any) -> None:
        try:
            if engine is None:
                raise ValueError("Embedding engine instance cannot be None.")
            if not hasattr(engine, "embed_chunks") or not callable(engine.embed_chunks):
                raise ValueError("Embedding engine must expose a callable embed_chunks method.")
            if int(getattr(engine, "embedding_dim", 0) or 0) <= 0:
                raise ValueError("Embedding engine must expose a positive embedding_dim.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | embedding engine validation failed: {exc}")
            raise EmbeddingStageError("Invalid embedding engine instance.", cause=exc) from exc

    def _validate_embedding_matrix(
        self,
        matrix: np.ndarray,
        row_count: int,
        config: EmbeddingConfig,
        scope: str,
    ) -> np.ndarray:
        try:
            array = self._ensure_numeric_matrix(
                matrix=matrix,
                row_count=row_count,
                output_dtype=config.output_dtype,
            )

            if array.size == 0:
                return array
            if int(array.shape[1]) <= 0:
                raise ValueError("Embedding output must contain at least one feature column.")
            if not np.isfinite(array).all():
                raise ValueError("Embedding output contains non-finite values.")
            if np.all(np.abs(array) <= config.zero_tolerance):
                raise ValueError(
                    f"Embedding output for '{scope}' is invalid because all vectors are zero within tolerance "
                    f"{config.zero_tolerance}."
                )
            if int(array.shape[0]) >= int(config.variance_check_min_rows):
                feature_variance = np.var(array, axis=0)
                if feature_variance.size > 0 and bool(np.all(feature_variance <= config.min_variance_threshold)):
                    raise ValueError(
                        f"Embedding output for '{scope}' is invalid because variance is below "
                        f"{config.min_variance_threshold} across all dimensions."
                    )
            return array
        except EmbeddingStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | embedding output validation failed for scope={scope}: {exc}")
            raise EmbeddingStageError("Invalid embedding output detected.", cause=exc) from exc

    def _validate_stage_output(self, result: EmbeddingResult) -> None:
        try:
            if not isinstance(result, EmbeddingResult):
                raise TypeError("Embedding stage output must be an EmbeddingResult instance.")
            if result.X_embedding is None:
                raise ValueError("Embedding stage returned no embedding matrix.")

            matrix = np.asarray(result.X_embedding)
            metadata = result.embedding_metadata or {}
            text_column_count = int(metadata.get("text_column_count", 0))

            if matrix.ndim != 2:
                raise ValueError("Embedding output must be a 2D matrix.")
            if not np.isfinite(matrix).all():
                raise ValueError("Embedding output contains non-finite values.")
            if text_column_count > 0 and int(matrix.shape[1]) <= 0:
                raise ValueError("Embedding stage produced no embedding features despite text input.")
            if int(metadata.get("row_count", matrix.shape[0])) != int(matrix.shape[0]):
                raise ValueError("Embedding metadata row_count does not match embedding output.")
            if int(metadata.get("embedding_dimension", matrix.shape[1])) != int(matrix.shape[1]):
                raise ValueError("Embedding metadata embedding_dimension does not match embedding output.")
            if metadata.get("model_name") != result.embedding_report.model_name:
                raise ValueError("Embedding metadata model_name does not match embedding report.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage output validation failed: {exc}")
            raise EmbeddingStageError("Invalid embedding stage output.", cause=exc) from exc

    def _canonicalize_merge_strategy(self, strategy: str) -> str:
        try:
            normalized = str(strategy).strip().lower().replace("-", "_")
            if normalized in self._merge_strategy_aliases:
                return self._merge_strategy_aliases[normalized]
            raise ValueError(f"Unsupported merge strategy '{strategy}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | merge strategy canonicalization failed: {exc}")
            raise EmbeddingStageError("Failed to canonicalize merge strategy.", cause=exc) from exc

    def _canonicalize_handle_empty(self, handle_empty: str) -> str:
        try:
            normalized = str(handle_empty).strip().lower().replace("-", "_")
            if normalized in self._handle_empty_aliases:
                return self._handle_empty_aliases[normalized]
            raise ValueError(f"Unsupported empty text handling mode '{handle_empty}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | empty text mode canonicalization failed: {exc}")
            raise EmbeddingStageError("Failed to canonicalize empty text handling.", cause=exc) from exc

    def _normalize_text_value(self, value: Any) -> str:
        try:
            if pd.isna(value):
                return ""
            return str(value).strip()
        except Exception:
            return str(value).strip()

    def _log(self, level: int, message: str) -> None:
        try:
            self.logger.log(level, f"[{self.stage_name}] - {message}")
        except Exception:
            logging.getLogger(__name__).log(level, f"[{self.stage_name}] - {message}")


def embed_text_features(
    X_text: Optional[pd.DataFrame],
    feature_map: Optional[Dict[str, Any]] = None,
    preprocessing_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[EmbeddingConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> EmbeddingResult:
    try:
        module = EmbeddingModule(config=config, logger=logger)
        return module.run(
            X_text=X_text,
            feature_map=feature_map,
            preprocessing_metadata=preprocessing_metadata,
        )
    except EmbeddingStageError:
        raise
    except Exception as exc:
        raise EmbeddingStageError("Unhandled embedding error.", cause=exc) from exc
