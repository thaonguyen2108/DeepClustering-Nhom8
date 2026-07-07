from __future__ import annotations

import importlib.util
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EvaluationConfig:
    compute_tsne: bool = True
    perplexity: float = 30.0
    random_state: Optional[int] = 42
    input_dtype: str = "float32"
    label_dtype: str = "int32"
    tsne_init: str = "random"
    tsne_metric: str = "euclidean"
    compute_davies_bouldin: bool = True
    compute_calinski_harabasz: bool = True


@dataclass
class EvaluationReport:
    stage: str
    row_count: int
    input_shape: Tuple[int, int]
    cluster_count: int
    silhouette_score: float
    davies_bouldin: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    cluster_distribution: Dict[int, int] = field(default_factory=dict)
    tsne_computed: bool = False
    tsne_shape: Optional[Tuple[int, int]] = None
    tsne_perplexity_used: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    evaluation_metrics: Dict[str, Optional[float]]
    cluster_distribution: Dict[int, int]
    visualization_data: Dict[str, Any]
    evaluation_report: EvaluationReport


class EvaluationStageError(RuntimeError):
    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        try:
            super().__init__(message)
            self.cause = cause
        except Exception as exc:
            raise RuntimeError("Failed to initialize EvaluationStageError.") from exc


class EvaluationModule:
    stage_name = "EVALUATION"
    _tsne_init_aliases = {
        "random": "random",
        "pca": "pca",
    }

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        try:
            self.config = config or EvaluationConfig()
            self.logger = logger or logging.getLogger(__name__)
        except Exception as exc:
            raise EvaluationStageError("Failed to initialize evaluation module.", cause=exc) from exc

    def run(
        self,
        Z_latent: Optional[Any],
        cluster_labels: Optional[Any],
        config: Optional[EvaluationConfig] = None,
    ) -> EvaluationResult:
        try:
            active_config = self._validate_config(config or self.config)
            latent_matrix, labels = self._validate_inputs(
                Z_latent=Z_latent,
                cluster_labels=cluster_labels,
                config=active_config,
            )
            row_count, latent_dim = int(latent_matrix.shape[0]), int(latent_matrix.shape[1])
            cluster_distribution = self._build_cluster_distribution(labels)
            cluster_count = int(len(cluster_distribution))

            self._log(
                logging.INFO,
                f"Stage start | rows={row_count} | input_shape={tuple(latent_matrix.shape)} | "
                f"clusters={cluster_count}",
            )

            evaluation_metrics = self._compute_metrics(
                latent_matrix=latent_matrix,
                labels=labels,
                config=active_config,
            )
            self._log(
                logging.INFO,
                f"Silhouette score | value={evaluation_metrics['silhouette_score']:.6f}",
            )
            self._log(logging.INFO, f"Number of clusters | value={cluster_count}")

            visualization_data, tsne_perplexity_used, warning_messages = self._build_visualization_data(
                latent_matrix=latent_matrix,
                labels=labels,
                config=active_config,
            )

            report = EvaluationReport(
                stage=self.stage_name,
                row_count=row_count,
                input_shape=(row_count, latent_dim),
                cluster_count=cluster_count,
                silhouette_score=float(evaluation_metrics["silhouette_score"]),
                davies_bouldin=evaluation_metrics["davies_bouldin"],
                calinski_harabasz=evaluation_metrics["calinski_harabasz"],
                cluster_distribution=cluster_distribution,
                tsne_computed=visualization_data["Z_2d"] is not None,
                tsne_shape=None
                if visualization_data["Z_2d"] is None
                else (int(visualization_data["Z_2d"].shape[0]), int(visualization_data["Z_2d"].shape[1])),
                tsne_perplexity_used=tsne_perplexity_used,
                warnings=warning_messages,
            )
            result = EvaluationResult(
                evaluation_metrics=evaluation_metrics,
                cluster_distribution=cluster_distribution,
                visualization_data=visualization_data,
                evaluation_report=report,
            )
            self._validate_stage_output(result=result, row_count=row_count)
            return result
        except EvaluationStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage failed with {type(exc).__name__}: {exc}")
            raise EvaluationStageError("Evaluation stage failed.", cause=exc) from exc

    def _validate_config(self, config: EvaluationConfig) -> EvaluationConfig:
        try:
            normalized = EvaluationConfig(
                compute_tsne=bool(config.compute_tsne),
                perplexity=float(config.perplexity),
                random_state=None if config.random_state is None else int(config.random_state),
                input_dtype=str(np.dtype(config.input_dtype)),
                label_dtype=str(np.dtype(config.label_dtype)),
                tsne_init=self._canonicalize_tsne_init(config.tsne_init),
                tsne_metric=str(config.tsne_metric).strip() or "euclidean",
                compute_davies_bouldin=bool(config.compute_davies_bouldin),
                compute_calinski_harabasz=bool(config.compute_calinski_harabasz),
            )
            if normalized.perplexity <= 0:
                raise ValueError("perplexity must be greater than 0.")
            if np.dtype(normalized.label_dtype).kind not in {"i", "u"}:
                raise ValueError("label_dtype must be an integer dtype.")
            return normalized
        except Exception as exc:
            self._log(logging.ERROR, f"Error | invalid evaluation config: {exc}")
            raise EvaluationStageError("Invalid evaluation configuration.", cause=exc) from exc

    def _validate_inputs(
        self,
        Z_latent: Optional[Any],
        cluster_labels: Optional[Any],
        config: EvaluationConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if Z_latent is None:
                raise ValueError("Z_latent cannot be None.")
            if cluster_labels is None:
                raise ValueError("cluster_labels cannot be None.")

            if hasattr(Z_latent, "to_numpy"):
                latent_matrix = np.asarray(Z_latent.to_numpy(), dtype=np.dtype(config.input_dtype))
            else:
                latent_matrix = np.asarray(Z_latent, dtype=np.dtype(config.input_dtype))

            labels = np.asarray(cluster_labels, dtype=np.dtype(config.label_dtype))

            if latent_matrix.ndim == 1:
                if latent_matrix.size == 0:
                    latent_matrix = latent_matrix.reshape(0, 0)
                else:
                    latent_matrix = latent_matrix.reshape(-1, 1)

            if labels.ndim != 1:
                raise ValueError("cluster_labels must be a 1D array.")
            if latent_matrix.ndim != 2:
                raise ValueError("Z_latent must be a 2D numeric matrix.")
            if int(latent_matrix.shape[0]) <= 0:
                raise ValueError("Z_latent must contain at least one row.")
            if int(latent_matrix.shape[1]) <= 0:
                raise ValueError("Z_latent must contain at least one feature column.")
            if int(labels.shape[0]) != int(latent_matrix.shape[0]):
                raise ValueError(
                    f"cluster_labels length mismatch: expected {latent_matrix.shape[0]}, received {labels.shape[0]}."
                )
            if not np.isfinite(latent_matrix).all():
                raise ValueError("Z_latent contains NaN or infinite values.")
            if not np.isfinite(labels.astype(np.float64)).all():
                raise ValueError("cluster_labels contains NaN or infinite values.")

            unique_labels = np.unique(labels)
            if int(unique_labels.size) < 2:
                raise ValueError("Evaluation requires at least 2 distinct clusters; silhouette cannot be computed.")
            if int(unique_labels.size) >= int(latent_matrix.shape[0]):
                raise ValueError("Evaluation requires fewer clusters than samples to compute silhouette score.")

            return latent_matrix, labels
        except Exception as exc:
            self._log(logging.ERROR, f"Error | evaluation input validation failed: {exc}")
            raise EvaluationStageError("Invalid evaluation input.", cause=exc) from exc

    def _compute_metrics(
        self,
        latent_matrix: np.ndarray,
        labels: np.ndarray,
        config: EvaluationConfig,
    ) -> Dict[str, Optional[float]]:
        try:
            metrics_runtime = self._load_sklearn_runtime()

            silhouette_value = float(metrics_runtime["silhouette_score"](latent_matrix, labels))
            davies_bouldin_value: Optional[float] = None
            calinski_harabasz_value: Optional[float] = None

            if config.compute_davies_bouldin:
                davies_bouldin_value = float(metrics_runtime["davies_bouldin_score"](latent_matrix, labels))
            if config.compute_calinski_harabasz:
                calinski_harabasz_value = float(metrics_runtime["calinski_harabasz_score"](latent_matrix, labels))

            return {
                "silhouette_score": silhouette_value,
                "davies_bouldin": davies_bouldin_value,
                "calinski_harabasz": calinski_harabasz_value,
            }
        except EvaluationStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | metric computation failed: {exc}")
            raise EvaluationStageError("Failed to compute clustering evaluation metrics.", cause=exc) from exc

    def _build_cluster_distribution(self, labels: np.ndarray) -> Dict[int, int]:
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            return {int(label): int(count) for label, count in zip(unique_labels.tolist(), counts.tolist())}
        except Exception as exc:
            self._log(logging.ERROR, f"Error | cluster distribution computation failed: {exc}")
            raise EvaluationStageError("Failed to compute cluster distribution.", cause=exc) from exc

    def _build_visualization_data(
        self,
        latent_matrix: np.ndarray,
        labels: np.ndarray,
        config: EvaluationConfig,
    ) -> Tuple[Dict[str, Any], Optional[float], List[str]]:
        try:
            warning_messages: List[str] = []
            if not config.compute_tsne:
                return {"Z_2d": None, "labels": labels.copy()}, None, warning_messages

            effective_perplexity = self._resolve_tsne_perplexity(
                row_count=int(latent_matrix.shape[0]),
                requested_perplexity=float(config.perplexity),
            )
            if not np.isclose(effective_perplexity, float(config.perplexity)):
                warning_message = (
                    f"t-SNE perplexity was adjusted from {config.perplexity} to {effective_perplexity} "
                    "to satisfy the sample-size constraint."
                )
                warning_messages.append(warning_message)
                self._log(
                    logging.WARNING,
                    f"Fallback usage | scope=tsne | action=adjust_perplexity | value={effective_perplexity}",
                )

            metrics_runtime = self._load_sklearn_runtime()
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                tsne = metrics_runtime["TSNE"](
                    n_components=2,
                    perplexity=float(effective_perplexity),
                    random_state=config.random_state,
                    init=str(config.tsne_init),
                    metric=str(config.tsne_metric),
                )
                Z_2d = np.asarray(tsne.fit_transform(latent_matrix), dtype=np.dtype(config.input_dtype))

            for caught_warning in caught_warnings:
                warning_text = str(caught_warning.message)
                if warning_text:
                    warning_messages.append(warning_text)
                    self._log(logging.WARNING, f"t-SNE warning | message={warning_text}")

            if Z_2d.ndim != 2 or tuple(Z_2d.shape) != (int(latent_matrix.shape[0]), 2):
                raise ValueError("t-SNE output must have shape (n_samples, 2).")
            if not np.isfinite(Z_2d).all():
                raise ValueError("t-SNE output contains NaN or infinite values.")

            return {"Z_2d": Z_2d, "labels": labels.copy()}, float(effective_perplexity), warning_messages
        except EvaluationStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | visualization data generation failed: {exc}")
            raise EvaluationStageError("Failed to build visualization-ready evaluation output.", cause=exc) from exc

    def _resolve_tsne_perplexity(
        self,
        row_count: int,
        requested_perplexity: float,
    ) -> float:
        try:
            if row_count < 3:
                raise ValueError("t-SNE requires at least 3 samples.")

            max_valid_perplexity = max(1.0, float(row_count - 1))
            return min(float(requested_perplexity), max_valid_perplexity)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | t-SNE perplexity resolution failed: {exc}")
            raise EvaluationStageError("Failed to resolve a valid t-SNE perplexity.", cause=exc) from exc

    def _load_sklearn_runtime(self) -> Dict[str, Any]:
        try:
            if importlib.util.find_spec("sklearn") is None:
                raise ImportError("Missing required dependency 'scikit-learn'. Install it with `pip install scikit-learn`.")

            from sklearn.manifold import TSNE
            from sklearn.metrics import (
                calinski_harabasz_score,
                davies_bouldin_score,
                silhouette_score,
            )

            return {
                "TSNE": TSNE,
                "silhouette_score": silhouette_score,
                "davies_bouldin_score": davies_bouldin_score,
                "calinski_harabasz_score": calinski_harabasz_score,
            }
        except Exception as exc:
            self._log(logging.ERROR, f"Error | scikit-learn runtime unavailable: {exc}")
            raise EvaluationStageError(
                "Evaluation stage requires scikit-learn. Install it with `pip install scikit-learn`.",
                cause=exc,
            ) from exc

    def _validate_stage_output(
        self,
        result: EvaluationResult,
        row_count: int,
    ) -> None:
        try:
            if not isinstance(result, EvaluationResult):
                raise TypeError("Evaluation stage output must be an EvaluationResult instance.")

            metrics = result.evaluation_metrics or {}
            if "silhouette_score" not in metrics or metrics["silhouette_score"] is None:
                raise ValueError("evaluation_metrics must contain silhouette_score.")
            if not np.isfinite(float(metrics["silhouette_score"])):
                raise ValueError("silhouette_score must be finite.")
            if metrics.get("davies_bouldin") is not None and not np.isfinite(float(metrics["davies_bouldin"])):
                raise ValueError("davies_bouldin must be finite when provided.")
            if metrics.get("calinski_harabasz") is not None and not np.isfinite(float(metrics["calinski_harabasz"])):
                raise ValueError("calinski_harabasz must be finite when provided.")

            distribution = result.cluster_distribution or {}
            if not distribution:
                raise ValueError("cluster_distribution cannot be empty.")
            if sum(int(count) for count in distribution.values()) != int(row_count):
                raise ValueError("cluster_distribution counts must sum to the row count.")

            visualization_data = result.visualization_data or {}
            labels = np.asarray(visualization_data.get("labels"))
            if labels.ndim != 1 or int(labels.shape[0]) != int(row_count):
                raise ValueError("visualization_data.labels must align with the row count.")

            Z_2d = visualization_data.get("Z_2d")
            if Z_2d is not None:
                projected = np.asarray(Z_2d)
                if projected.ndim != 2 or tuple(projected.shape) != (int(row_count), 2):
                    raise ValueError("visualization_data.Z_2d must have shape (n_samples, 2) when provided.")
                if not np.isfinite(projected).all():
                    raise ValueError("visualization_data.Z_2d contains non-finite values.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage output validation failed: {exc}")
            raise EvaluationStageError("Invalid evaluation stage output.", cause=exc) from exc

    def _canonicalize_tsne_init(self, init: str) -> str:
        try:
            normalized = str(init).strip().lower()
            if normalized in self._tsne_init_aliases:
                return self._tsne_init_aliases[normalized]
            raise ValueError(f"Unsupported t-SNE init strategy '{init}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | t-SNE init canonicalization failed: {exc}")
            raise EvaluationStageError("Failed to canonicalize t-SNE init strategy.", cause=exc) from exc

    def _log(self, level: int, message: str) -> None:
        try:
            self.logger.log(level, f"[{self.stage_name}] - {message}")
        except Exception:
            logging.getLogger(__name__).log(level, f"[{self.stage_name}] - {message}")


def evaluate_clustering(
    Z_latent: Optional[Any],
    cluster_labels: Optional[Any],
    config: Optional[EvaluationConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> EvaluationResult:
    try:
        module = EvaluationModule(config=config, logger=logger)
        return module.run(Z_latent=Z_latent, cluster_labels=cluster_labels)
    except EvaluationStageError:
        raise
    except Exception as exc:
        raise EvaluationStageError("Unhandled evaluation error.", cause=exc) from exc
