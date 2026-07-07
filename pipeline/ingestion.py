from __future__ import annotations

import importlib.util
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, TextIO, Union

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


TabularSource = Union[str, Path, bytes, bytearray, BinaryIO, TextIO]


@dataclass
class IngestionConfig:
    file_format: Optional[str] = None
    nrows: Optional[int] = None
    csv_encoding: Optional[str] = None
    csv_encoding_candidates: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    csv_separator: Optional[str] = None
    csv_delimiter_candidates: tuple[str, ...] = (",", ";", "\t", "|")
    csv_engine: Optional[str] = None
    csv_low_memory: bool = False
    csv_quotechar: str = '"'
    csv_escapechar: Optional[str] = None
    csv_reader_kwargs: Dict[str, Any] = field(default_factory=dict)
    excel_sheet_name: Union[int, str, None] = 0
    excel_engine: Optional[str] = None
    excel_engine_candidates: tuple[str, ...] = ("openpyxl",)
    excel_reader_kwargs: Dict[str, Any] = field(default_factory=dict)
    json_lines: Optional[bool] = None
    json_orient: Optional[str] = None
    json_encoding: str = "utf-8"
    json_reader_kwargs: Dict[str, Any] = field(default_factory=dict)
    stringify_column_names: bool = True
    trim_column_names: bool = True
    fill_empty_column_names: bool = True
    empty_column_prefix: str = "unnamed_column"
    deduplicate_column_names: bool = True
    duplicate_column_separator: str = "__"
    drop_empty_rows: bool = False
    allow_empty_frame: bool = False
    include_null_counts: bool = True
    include_preview: bool = True
    sample_rows: int = 5


@dataclass
class PreparedSource:
    source_name: str
    source_type: str
    path: Optional[Path] = None
    payload: Optional[bytes] = None

    def open_binary(self) -> Union[str, io.BytesIO]:
        try:
            if self.path is not None:
                return str(self.path)
            if self.payload is None:
                raise ValueError("No payload available for in-memory source.")
            return io.BytesIO(self.payload)
        except Exception as exc:
            raise ValueError(f"Unable to open binary source '{self.source_name}'.") from exc

    def open_text(self, encoding: str = "utf-8", errors: str = "strict") -> Union[str, io.StringIO]:
        try:
            if self.path is not None:
                return str(self.path)
            if self.payload is None:
                raise ValueError("No payload available for in-memory source.")
            return io.StringIO(self.payload.decode(encoding, errors=errors))
        except Exception as exc:
            raise ValueError(f"Unable to open text source '{self.source_name}'.") from exc

    def head_bytes(self, size: int = 256) -> bytes:
        try:
            if self.path is not None:
                with self.path.open("rb") as file_handle:
                    return file_handle.read(size)
            return (self.payload or b"")[:size]
        except Exception as exc:
            raise ValueError(f"Unable to read header bytes from '{self.source_name}'.") from exc


@dataclass
class IngestionResult:
    dataframe: pd.DataFrame
    metadata: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


class IngestionStageError(RuntimeError):
    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        try:
            super().__init__(message)
            self.cause = cause
        except Exception as exc:
            raise RuntimeError("Failed to initialize IngestionStageError.") from exc


class IngestionModule:
    stage_name = "INGESTION"
    _supported_formats = {"csv", "json", "excel"}

    def __init__(self, config: Optional[IngestionConfig] = None, logger: Optional[logging.Logger] = None):
        try:
            self.config = config or IngestionConfig()
            self.logger = logger or logging.getLogger(__name__)
        except Exception as exc:
            raise IngestionStageError("Failed to initialize ingestion module.", cause=exc) from exc

    def run(self, source: TabularSource, config: Optional[IngestionConfig] = None) -> IngestionResult:
        try:
            active_config = config or self.config
            self._log(
                logging.INFO,
                f"Stage start | requested_format={active_config.file_format or 'auto'} | "
                f"nrows={active_config.nrows or 'all'}",
            )

            prepared_source = self._prepare_source(source, active_config)
            resolved_format = self._resolve_file_format(prepared_source, active_config)
            self._log(
                logging.INFO,
                f"Input summary | source={prepared_source.source_name} | "
                f"source_type={prepared_source.source_type} | detected_format={resolved_format}",
            )

            dataframe, load_details, warnings = self._load_dataframe(prepared_source, resolved_format, active_config)
            normalized_dataframe, normalization_warnings = self._normalize_dataframe(dataframe, active_config)
            warnings.extend(normalization_warnings)

            metadata = self._build_metadata(
                prepared_source=prepared_source,
                dataframe=normalized_dataframe,
                resolved_format=resolved_format,
                config=active_config,
                load_details=load_details,
                warnings=warnings,
            )

            self._log(
                logging.INFO,
                f"Output summary | rows={metadata['row_count']} | columns={metadata['column_count']} | "
                f"warnings={len(warnings)}",
            )

            result = IngestionResult(dataframe=normalized_dataframe, metadata=metadata, warnings=warnings)
            self._validate_stage_output(result)
            return result
        except IngestionStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage failed with {type(exc).__name__}: {exc}")
            raise IngestionStageError("Dataset ingestion failed.", cause=exc) from exc

    def _prepare_source(self, source: TabularSource, config: IngestionConfig) -> PreparedSource:
        try:
            if isinstance(source, Path):
                path = source.expanduser().resolve()
                if not path.exists():
                    raise FileNotFoundError(f"Input file does not exist: {path}")
                return PreparedSource(source_name=path.name, source_type="path", path=path)

            if isinstance(source, str):
                candidate_path = Path(source).expanduser()
                if candidate_path.exists():
                    path = candidate_path.resolve()
                    return PreparedSource(source_name=path.name, source_type="path", path=path)
                if config.file_format is not None:
                    payload = source.encode(config.json_encoding)
                    inline_name = f"inline_input.{self._display_format(config.file_format)}"
                    return PreparedSource(source_name=inline_name, source_type="inline_text", payload=payload)
                raise FileNotFoundError(f"Input path does not exist: {source}")

            if isinstance(source, (bytes, bytearray)):
                source_name = f"in_memory_dataset.{self._display_format(config.file_format or 'csv')}"
                return PreparedSource(source_name=source_name, source_type="bytes", payload=bytes(source))

            if hasattr(source, "read"):
                if hasattr(source, "seek"):
                    source.seek(0)
                payload = source.read()
                if hasattr(source, "seek"):
                    source.seek(0)
                if isinstance(payload, str):
                    payload = payload.encode(config.json_encoding)
                if not isinstance(payload, (bytes, bytearray)):
                    raise TypeError("Readable source must return bytes or text.")
                source_name = Path(getattr(source, "name", "")).name or (
                    f"in_memory_dataset.{self._display_format(config.file_format or 'csv')}"
                )
                return PreparedSource(source_name=source_name, source_type="file_like", payload=bytes(payload))

            raise TypeError(f"Unsupported source type: {type(source).__name__}")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | source preparation failed: {exc}")
            raise IngestionStageError("Failed to prepare ingestion source.", cause=exc) from exc

    def _resolve_file_format(self, prepared_source: PreparedSource, config: IngestionConfig) -> str:
        try:
            if config.file_format:
                normalized_format = self._normalize_format(config.file_format)
                self._log(logging.INFO, f"Input summary | using configured format={normalized_format}")
                return normalized_format

            suffix = Path(prepared_source.source_name).suffix.lower()
            suffix_map = {
                ".csv": "csv",
                ".json": "json",
                ".jsonl": "json",
                ".ndjson": "json",
                ".xlsx": "excel",
                ".xls": "excel",
            }
            if suffix in suffix_map:
                return suffix_map[suffix]

            header_bytes = prepared_source.head_bytes()
            stripped_header = header_bytes.lstrip()
            if stripped_header.startswith((b"{", b"[")):
                return "json"
            if stripped_header.startswith(b"PK"):
                return "excel"
            return "csv"
        except Exception as exc:
            self._log(logging.ERROR, f"Error | format detection failed: {exc}")
            raise IngestionStageError("Failed to resolve file format.", cause=exc) from exc

    def _load_dataframe(
        self,
        prepared_source: PreparedSource,
        resolved_format: str,
        config: IngestionConfig,
    ) -> tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        try:
            if resolved_format == "csv":
                return self._load_csv(prepared_source, config)
            if resolved_format == "excel":
                return self._load_excel(prepared_source, config)
            if resolved_format == "json":
                return self._load_json(prepared_source, config)
            raise ValueError(f"Unsupported file format: {resolved_format}")
        except IngestionStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | reader failed for format={resolved_format}: {exc}")
            raise IngestionStageError("Failed to load dataset into DataFrame.", cause=exc) from exc

    def _load_csv(
        self,
        prepared_source: PreparedSource,
        config: IngestionConfig,
    ) -> tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        try:
            warnings: List[str] = []
            encodings = self._ordered_candidates(config.csv_encoding, config.csv_encoding_candidates)
            separators: Sequence[Optional[str]] = (
                [config.csv_separator]
                if config.csv_separator is not None
                else [None, *config.csv_delimiter_candidates]
            )

            attempts: List[str] = []
            for encoding in encodings:
                for separator in separators:
                    read_kwargs = dict(config.csv_reader_kwargs)
                    engine = config.csv_engine or ("python" if separator is None else None)
                    if engine is not None:
                        read_kwargs["engine"] = engine

                    try:
                        dataframe = pd.read_csv(
                            prepared_source.open_binary(),
                            encoding=encoding,
                            sep=separator,
                            nrows=config.nrows,
                            low_memory=config.csv_low_memory,
                            quotechar=config.csv_quotechar,
                            escapechar=config.csv_escapechar,
                            **read_kwargs,
                        )
                        if separator is not None:
                            warnings.append(f"CSV loaded with explicit separator '{separator}'.")
                        return (
                            dataframe,
                            {
                                "reader": "csv",
                                "encoding": encoding,
                                "separator": "auto" if separator is None else separator,
                            },
                            warnings,
                        )
                    except (UnicodeDecodeError, ParserError, EmptyDataError, ValueError) as exc:
                        attempts.append(
                            f"encoding={encoding}, separator={'auto' if separator is None else separator}: "
                            f"{type(exc).__name__}"
                        )

            attempt_summary = "; ".join(attempts) if attempts else "no reader attempts recorded"
            raise ValueError(
                "CSV ingestion failed after trying all encoding and separator candidates. "
                f"Attempts: {attempt_summary}"
            )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | csv reader failed: {exc}")
            raise IngestionStageError("CSV ingestion failed.", cause=exc) from exc

    def _load_excel(
        self,
        prepared_source: PreparedSource,
        config: IngestionConfig,
    ) -> tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        try:
            warnings: List[str] = []
            read_kwargs = dict(config.excel_reader_kwargs)
            read_kwargs["engine"] = self._resolve_excel_engine(config)

            dataframe = pd.read_excel(
                prepared_source.open_binary(),
                sheet_name=config.excel_sheet_name,
                nrows=config.nrows,
                **read_kwargs,
            )

            selected_sheet = config.excel_sheet_name
            if isinstance(dataframe, dict):
                if not dataframe:
                    raise ValueError("Excel source did not contain readable sheets.")

                selected_sheet = next(iter(dataframe))
                selected_frame = dataframe[selected_sheet]
                for sheet_name, sheet_frame in dataframe.items():
                    if not sheet_frame.empty:
                        selected_sheet = sheet_name
                        selected_frame = sheet_frame
                        break
                dataframe = selected_frame
                warnings.append("Excel workbook returned multiple sheets; selected the first non-empty sheet.")

            return (
                dataframe,
                {"reader": "excel", "sheet_name": selected_sheet},
                warnings,
            )
        except IngestionStageError:
            raise
        except ImportError as exc:
            message = (
                "Excel ingestion requires openpyxl or another installed Excel engine. "
                "Install it with `pip install openpyxl` or set excel_engine to an available backend."
            )
            self._log(logging.ERROR, f"Error | excel dependency missing: {exc}")
            raise IngestionStageError(message, cause=exc) from exc
        except Exception as exc:
            self._log(logging.ERROR, f"Error | excel reader failed: {exc}")
            raise IngestionStageError("Excel ingestion failed.", cause=exc) from exc

    def _load_json(
        self,
        prepared_source: PreparedSource,
        config: IngestionConfig,
    ) -> tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        try:
            warnings: List[str] = []
            line_modes: Sequence[bool] = (
                [config.json_lines] if config.json_lines is not None else [False, True]
            )

            for lines in line_modes:
                read_kwargs = dict(config.json_reader_kwargs)
                if config.json_orient is not None:
                    read_kwargs["orient"] = config.json_orient

                try:
                    dataframe = pd.read_json(
                        prepared_source.open_text(encoding=config.json_encoding, errors="replace"),
                        lines=lines,
                        **read_kwargs,
                    )
                    if config.nrows is not None:
                        dataframe = dataframe.head(config.nrows)
                    if config.json_lines is None and lines:
                        warnings.append("JSON loaded with lines=True fallback.")
                    return (
                        dataframe,
                        {"reader": "json", "lines": lines, "orient": config.json_orient or "auto"},
                        warnings,
                    )
                except ValueError:
                    continue

            raise ValueError("JSON ingestion failed for both standard JSON and line-delimited JSON modes.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | json reader failed: {exc}")
            raise IngestionStageError("JSON ingestion failed.", cause=exc) from exc

    def _normalize_dataframe(
        self,
        dataframe: pd.DataFrame,
        config: IngestionConfig,
    ) -> tuple[pd.DataFrame, List[str]]:
        try:
            warnings: List[str] = []
            normalized = dataframe.copy()

            if config.drop_empty_rows:
                before_count = len(normalized)
                normalized = normalized.dropna(how="all")
                dropped_rows = before_count - len(normalized)
                if dropped_rows > 0:
                    warnings.append(f"Dropped {dropped_rows} fully empty rows during ingestion.")

            updated_columns = self._normalize_columns(list(normalized.columns), config)
            if list(normalized.columns) != updated_columns:
                normalized.columns = updated_columns
                warnings.append("Column names were normalized during ingestion.")

            normalized = normalized.reset_index(drop=True)

            if normalized.empty and not config.allow_empty_frame:
                raise ValueError("Loaded dataset is empty after ingestion.")
            if normalized.shape[1] == 0:
                raise ValueError("Loaded dataset does not contain columns.")

            return normalized, warnings
        except Exception as exc:
            self._log(logging.ERROR, f"Error | dataframe normalization failed: {exc}")
            raise IngestionStageError("Failed to normalize ingested DataFrame.", cause=exc) from exc

    def _normalize_columns(self, columns: List[Any], config: IngestionConfig) -> List[str]:
        try:
            normalized_columns: List[str] = []
            for index, column in enumerate(columns):
                column_name = column
                if config.stringify_column_names:
                    column_name = str(column_name)
                if config.trim_column_names:
                    column_name = str(column_name).strip()
                if not isinstance(column_name, str):
                    column_name = str(column_name)
                if config.fill_empty_column_names and column_name == "":
                    column_name = f"{config.empty_column_prefix}_{index}"
                normalized_columns.append(column_name)

            if config.deduplicate_column_names:
                return self._deduplicate_columns(normalized_columns, config.duplicate_column_separator)
            return normalized_columns
        except Exception as exc:
            self._log(logging.ERROR, f"Error | column normalization failed: {exc}")
            raise IngestionStageError("Failed to normalize column names.", cause=exc) from exc

    def _deduplicate_columns(self, columns: List[str], separator: str) -> List[str]:
        try:
            counts: Dict[str, int] = {}
            deduplicated: List[str] = []

            for column in columns:
                next_index = counts.get(column, 0)
                candidate = column if next_index == 0 else f"{column}{separator}{next_index}"
                while candidate in deduplicated:
                    next_index += 1
                    candidate = f"{column}{separator}{next_index}"
                counts[column] = next_index + 1
                deduplicated.append(candidate)

            return deduplicated
        except Exception as exc:
            self._log(logging.ERROR, f"Error | duplicate column handling failed: {exc}")
            raise IngestionStageError("Failed to deduplicate column names.", cause=exc) from exc

    def _build_metadata(
        self,
        prepared_source: PreparedSource,
        dataframe: pd.DataFrame,
        resolved_format: str,
        config: IngestionConfig,
        load_details: Dict[str, Any],
        warnings: List[str],
    ) -> Dict[str, Any]:
        try:
            metadata: Dict[str, Any] = {
                "stage": self.stage_name,
                "source_name": prepared_source.source_name,
                "source_type": prepared_source.source_type,
                "source_path": str(prepared_source.path) if prepared_source.path is not None else None,
                "file_format": resolved_format,
                "row_count": int(dataframe.shape[0]),
                "column_count": int(dataframe.shape[1]),
                "columns": list(dataframe.columns),
                "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
                "memory_usage_bytes": int(dataframe.memory_usage(deep=True).sum()),
                "load_details": load_details,
                "warnings": list(warnings),
            }

            if config.include_null_counts:
                metadata["null_counts"] = {
                    column: int(count) for column, count in dataframe.isna().sum().to_dict().items()
                }

            if config.include_preview and config.sample_rows > 0:
                metadata["preview"] = self._preview_records(dataframe, config.sample_rows)

            return metadata
        except Exception as exc:
            self._log(logging.ERROR, f"Error | metadata construction failed: {exc}")
            raise IngestionStageError("Failed to build ingestion metadata.", cause=exc) from exc

    def _preview_records(self, dataframe: pd.DataFrame, sample_rows: int) -> List[Dict[str, Any]]:
        try:
            preview_frame = dataframe.head(sample_rows)
            preview_records: List[Dict[str, Any]] = []

            for _, row in preview_frame.iterrows():
                serialized_row = {
                    str(column): self._serialize_preview_value(value) for column, value in row.items()
                }
                preview_records.append(serialized_row)

            return preview_records
        except Exception as exc:
            self._log(logging.ERROR, f"Error | preview generation failed: {exc}")
            raise IngestionStageError("Failed to create dataset preview.", cause=exc) from exc

    def _serialize_preview_value(self, value: Any) -> Any:
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

    def _ordered_candidates(
        self,
        preferred_value: Optional[str],
        fallback_values: Sequence[str],
    ) -> List[str]:
        try:
            candidates: List[str] = []
            if preferred_value:
                candidates.append(preferred_value)
            for value in fallback_values:
                if value not in candidates:
                    candidates.append(value)
            return candidates
        except Exception as exc:
            self._log(logging.ERROR, f"Error | candidate ordering failed: {exc}")
            raise IngestionStageError("Failed to build fallback candidates.", cause=exc) from exc

    def _resolve_excel_engine(self, config: IngestionConfig) -> str:
        try:
            if config.excel_engine is not None:
                if importlib.util.find_spec(config.excel_engine) is None:
                    raise ImportError(f"Configured Excel engine '{config.excel_engine}' is not installed.")
                return config.excel_engine

            for engine_name in config.excel_engine_candidates:
                if importlib.util.find_spec(engine_name) is not None:
                    return engine_name

            raise ImportError(
                "No Excel engine is installed. Install openpyxl or provide an available excel_engine."
            )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | excel engine resolution failed: {exc}")
            message = (
                "Excel ingestion requires openpyxl or another installed Excel engine. "
                "Install it with `pip install openpyxl` or set excel_engine to an available backend."
            )
            raise IngestionStageError(message, cause=exc) from exc

    def _validate_stage_output(self, result: IngestionResult) -> None:
        try:
            if not isinstance(result, IngestionResult):
                raise TypeError("Ingestion stage output must be an IngestionResult instance.")
            if not isinstance(result.dataframe, pd.DataFrame):
                raise TypeError("Ingestion stage must return a pandas DataFrame.")

            dataframe = result.dataframe
            metadata = result.metadata or {}

            if dataframe.ndim != 2:
                raise ValueError("Ingestion output DataFrame must be 2-dimensional.")
            if int(dataframe.shape[1]) <= 0:
                raise ValueError("Ingestion output DataFrame must contain at least one column.")

            expected_rows = metadata.get("row_count")
            expected_columns = metadata.get("column_count")
            if expected_rows is not None and int(expected_rows) != int(dataframe.shape[0]):
                raise ValueError(
                    f"Ingestion metadata row_count mismatch: expected {expected_rows}, got {dataframe.shape[0]}."
                )
            if expected_columns is not None and int(expected_columns) != int(dataframe.shape[1]):
                raise ValueError(
                    "Ingestion metadata column_count mismatch: "
                    f"expected {expected_columns}, got {dataframe.shape[1]}."
                )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | output validation failed: {exc}")
            raise IngestionStageError("Invalid ingestion stage output.", cause=exc) from exc

    def _normalize_format(self, file_format: str) -> str:
        try:
            normalized = file_format.strip().lower()
            alias_map = {
                "csv": "csv",
                "json": "json",
                "jsonl": "json",
                "ndjson": "json",
                "xlsx": "excel",
                "xls": "excel",
                "excel": "excel",
            }
            if normalized not in alias_map:
                raise ValueError(
                    "Unsupported file format. Expected one of: csv, json, xlsx, xls, excel."
                )
            resolved = alias_map[normalized]
            if resolved not in self._supported_formats:
                raise ValueError(f"Unsupported canonical format: {resolved}")
            return resolved
        except Exception as exc:
            self._log(logging.ERROR, f"Error | format normalization failed: {exc}")
            raise IngestionStageError("Failed to normalize file format.", cause=exc) from exc

    def _display_format(self, file_format: str) -> str:
        try:
            normalized = file_format.strip().lower()
            if normalized in {"xlsx", "xls", "excel"}:
                return "xlsx"
            if normalized in {"json", "jsonl", "ndjson"}:
                return "json"
            return "csv"
        except Exception:
            return "csv"

    def _log(self, level: int, message: str) -> None:
        try:
            self.logger.log(level, f"[{self.stage_name}] - {message}")
        except Exception:
            logging.getLogger(__name__).log(level, f"[{self.stage_name}] - {message}")


def load_tabular_dataset(
    source: TabularSource,
    config: Optional[IngestionConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> IngestionResult:
    try:
        module = IngestionModule(config=config, logger=logger)
        return module.run(source)
    except IngestionStageError:
        raise
    except Exception as exc:
        raise IngestionStageError("Unhandled ingestion error.", cause=exc) from exc
