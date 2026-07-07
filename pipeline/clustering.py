from __future__ import annotations

import importlib.util
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ClusteringConfig:
    k_min: int = 2
    k_max: int = 10
    random_state: Optional[int] = 42
    n_init: int = 10
    max_iter: int = 300
    algorithm: str = "lloyd"
    silhouette_metric: str = "euclidean"
    silhouette_sample_size: Optional[int] = None
    prefer_elbow_when_close: bool = True
    elbow_validation_tolerance: float = 0.02
    input_dtype: str = "float32"
    label_dtype: str = "int32"


@dataclass
class ClusteringReport:
    stage: str
    row_count: int
    input_shape: Tuple[int, int]
    tested_k_values: List[int] = field(default_factory=list)
    silhouette_scores: Dict[int, float] = field(default_factory=dict)
    inertia_values: Dict[int, float] = field(default_factory=dict)
    best_k: int = 0
    selected_k: int = 0
    silhouette_best_k: int = 0
    elbow_k: Optional[int] = None
    selected_silhouette_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class ClusteringResult:
    cluster_labels: np.ndarray
    best_k: int
    clustering_metadata: Dict[str, Any]
    clustering_report: ClusteringReport


class ClusteringStageError(RuntimeError):
    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        try:
            super().__init__(message)
            self.cause = cause
        except Exception as exc:
            raise RuntimeError("Failed to initialize ClusteringStageError.") from exc


class ClusteringModule:
    stage_name = "CLUSTERING"
    _algorithm_aliases = {
        "lloyd": "lloyd",
        "elkan": "elkan",
        "full": "lloyd",
        "auto": "lloyd",
    }

    def __init__(
        self,
        config: Optional[ClusteringConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        try:
            self.config = config or ClusteringConfig()
            self.logger = logger or logging.getLogger(__name__)
        except Exception as exc:
            raise ClusteringStageError("Failed to initialize clustering module.", cause=exc) from exc

    def run(
        self,
        Z_latent: Optional[Any],
        config: Optional[ClusteringConfig] = None,
    ) -> ClusteringResult:
        try:
            active_config = self._validate_config(config or self.config)
            latent_matrix = self._validate_input_matrix(Z_latent=Z_latent, config=active_config)
            row_count, input_dim = int(latent_matrix.shape[0]), int(latent_matrix.shape[1])

            self._log(logging.INFO, f"Stage start | rows={row_count} | input_shape={tuple(latent_matrix.shape)}")

            candidate_k_values = self._resolve_candidate_k_values(row_count=row_count, config=active_config)
            self._log(logging.INFO, f"Tested K values | candidates={candidate_k_values}")

            silhouette_scores, inertia_values, warnings_list = self._evaluate_k_candidates(
                latent_matrix=latent_matrix,
                candidate_k_values=candidate_k_values,
                config=active_config,
            )
            tested_k_values = sorted(silhouette_scores.keys())
            if not tested_k_values:
                raise ValueError("No valid K values produced successful clustering results.")

            best_k, silhouette_best_k, elbow_k = self._select_best_k(
                tested_k_values=tested_k_values,
                silhouette_scores=silhouette_scores,
                inertia_values=inertia_values,
                config=active_config,
            )

            cluster_labels, selected_inertia, selected_silhouette = self._fit_selected_k(
                latent_matrix=latent_matrix,
                selected_k=best_k,
                config=active_config,
            )
            cluster_labels = self._validate_cluster_labels(
                cluster_labels=cluster_labels,
                row_count=row_count,
                best_k=best_k,
                config=active_config,
            )

            self._log(
                logging.INFO,
                f"Best K selected | best_k={best_k} | silhouette={selected_silhouette:.6f} | elbow_k={elbow_k}",
            )

            report = ClusteringReport(
                stage=self.stage_name,
                row_count=row_count,
                input_shape=(row_count, input_dim),
                tested_k_values=tested_k_values,
                silhouette_scores={int(k): float(silhouette_scores[k]) for k in tested_k_values},
                inertia_values={int(k): float(inertia_values[k]) for k in tested_k_values},
                best_k=int(best_k),
                selected_k=int(best_k),
                silhouette_best_k=int(silhouette_best_k),
                elbow_k=None if elbow_k is None else int(elbow_k),
                selected_silhouette_score=float(selected_silhouette),
                warnings=warnings_list,
            )
            metadata = self._build_clustering_metadata(
                report=report,
                cluster_labels=cluster_labels,
                selected_inertia=selected_inertia,
            )
            result = ClusteringResult(
                cluster_labels=cluster_labels,
                best_k=int(best_k),
                clustering_metadata=metadata,
                clustering_report=report,
            )
            self._validate_stage_output(result)
            return result
        except ClusteringStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage failed with {type(exc).__name__}: {exc}")
            raise ClusteringStageError("Clustering stage failed.", cause=exc) from exc

    def _validate_config(self, config: ClusteringConfig) -> ClusteringConfig:
        try:
            normalized = ClusteringConfig(
                k_min=max(2, int(config.k_min)),
                k_max=max(2, int(config.k_max)),
                random_state=None if config.random_state is None else int(config.random_state),
                n_init=max(1, int(config.n_init)),
                max_iter=max(1, int(config.max_iter)),
                algorithm=self._canonicalize_algorithm(config.algorithm),
                silhouette_metric=str(config.silhouette_metric).strip() or "euclidean",
                silhouette_sample_size=(
                    None if config.silhouette_sample_size is None else max(2, int(config.silhouette_sample_size))
                ),
                prefer_elbow_when_close=bool(config.prefer_elbow_when_close),
                elbow_validation_tolerance=max(0.0, float(config.elbow_validation_tolerance)),
                input_dtype=str(np.dtype(config.input_dtype)),
                label_dtype=str(np.dtype(config.label_dtype)),
            )
            if normalized.k_max < normalized.k_min:
                raise ValueError("k_max must be greater than or equal to k_min.")
            if np.dtype(normalized.label_dtype).kind not in {"i", "u"}:
                raise ValueError("label_dtype must be an integer dtype.")
            return normalized
        except Exception as exc:
            self._log(logging.ERROR, f"Error | invalid clustering config: {exc}")
            raise ClusteringStageError("Invalid clustering configuration.", cause=exc) from exc

    def _validate_input_matrix(
        self,
        Z_latent: Optional[Any],
        config: ClusteringConfig,
    ) -> np.ndarray:
        try:
            if Z_latent is None:
                raise ValueError("Z_latent cannot be None.")

            if hasattr(Z_latent, "to_numpy"):
                latent_matrix = np.asarray(Z_latent.to_numpy(), dtype=np.dtype(config.input_dtype))
            else:
                latent_matrix = np.asarray(Z_latent, dtype=np.dtype(config.input_dtype))

            if latent_matrix.ndim == 1:
                if latent_matrix.size == 0:
                    latent_matrix = latent_matrix.reshape(0, 0)
                else:
                    latent_matrix = latent_matrix.reshape(-1, 1)

            if latent_matrix.ndim != 2:
                raise ValueError("Z_latent must be a 2D numeric matrix.")
            if int(latent_matrix.shape[0]) < 3:
                raise ValueError("Clustering requires at least 3 samples to evaluate silhouette scores.")
            if int(latent_matrix.shape[1]) <= 0:
                raise ValueError("Z_latent must contain at least one feature column.")
            if not np.isfinite(latent_matrix).all():
                raise ValueError("Z_latent contains NaN or infinite values.")

            return latent_matrix
        except Exception as exc:
            self._log(logging.ERROR, f"Error | latent input validation failed: {exc}")
            raise ClusteringStageError("Invalid clustering input.", cause=exc) from exc

    def _resolve_candidate_k_values(
        self,
        row_count: int,
        config: ClusteringConfig,
    ) -> List[int]:
        try:
            max_valid_k = min(int(config.k_max), int(row_count) - 1)
            min_valid_k = max(2, int(config.k_min))
            if max_valid_k < min_valid_k:
                raise ValueError(
                    f"No valid K values available for row_count={row_count}. "
                    "Silhouette scoring requires 2 <= k < n_samples."
                )
            return list(range(min_valid_k, max_valid_k + 1))
        except Exception as exc:
            self._log(logging.ERROR, f"Error | K-range resolution failed: {exc}")
            raise ClusteringStageError("Failed to resolve valid K values for clustering.", cause=exc) from exc

    def _evaluate_k_candidates(
        self,
        latent_matrix: np.ndarray,
        candidate_k_values: Sequence[int],
        config: ClusteringConfig,
    ) -> Tuple[Dict[int, float], Dict[int, float], List[str]]:
        try:
            silhouette_scores: Dict[int, float] = {}
            inertia_values: Dict[int, float] = {}
            warnings_list: List[str] = []

            for k_value in candidate_k_values:
                try:
                    _, inertia_value, silhouette_value = self._fit_kmeans_and_score(
                        latent_matrix=latent_matrix,
                        k_value=int(k_value),
                        config=config,
                    )
                    silhouette_scores[int(k_value)] = float(silhouette_value)
                    inertia_values[int(k_value)] = float(inertia_value)
                    self._log(
                        logging.INFO,
                        f"K evaluation | k={k_value} | inertia={inertia_value:.6f} | "
                        f"silhouette={silhouette_value:.6f}",
                    )
                except Exception as exc:
                    warning_message = f"K={k_value} was skipped during clustering evaluation: {exc}"
                    warnings_list.append(warning_message)
                    self._log(logging.WARNING, f"Skipped case | k={k_value} | reason={exc}")

            return silhouette_scores, inertia_values, warnings_list
        except Exception as exc:
            self._log(logging.ERROR, f"Error | K evaluation failed: {exc}")
            raise ClusteringStageError("Failed to evaluate clustering candidates.", cause=exc) from exc

    def _select_best_k(
        self,
        tested_k_values: Sequence[int],
        silhouette_scores: Dict[int, float],
        inertia_values: Dict[int, float],
        config: ClusteringConfig,
    ) -> Tuple[int, int, Optional[int]]:
        try:
            silhouette_best_k = max(
                tested_k_values,
                key=lambda k_value: (float(silhouette_scores[int(k_value)]), -int(k_value)),
            )
            selected_k = int(silhouette_best_k)
            elbow_k = self._estimate_elbow_k(tested_k_values=tested_k_values, inertia_values=inertia_values)

            if config.prefer_elbow_when_close and elbow_k is not None and int(elbow_k) in silhouette_scores:
                silhouette_gap = float(silhouette_scores[int(silhouette_best_k)]) - float(
                    silhouette_scores[int(elbow_k)]
                )
                if silhouette_gap <= float(config.elbow_validation_tolerance):
                    selected_k = int(elbow_k)

            return int(selected_k), int(silhouette_best_k), None if elbow_k is None else int(elbow_k)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | best-K selection failed: {exc}")
            raise ClusteringStageError("Failed to select the best K value.", cause=exc) from exc

    def _estimate_elbow_k(
        self,
        tested_k_values: Sequence[int],
        inertia_values: Dict[int, float],
    ) -> Optional[int]:
        try:
            if len(tested_k_values) < 3:
                return None

            x_values = np.asarray([int(k_value) for k_value in tested_k_values], dtype=np.float64)
            y_values = np.asarray([float(inertia_values[int(k_value)]) for k_value in tested_k_values], dtype=np.float64)

            if not np.isfinite(y_values).all():
                raise ValueError("Inertia values contain non-finite values.")
            if np.allclose(y_values, y_values[0]):
                return int(tested_k_values[0])

            x_span = float(x_values[-1] - x_values[0])
            y_span = float(y_values.max() - y_values.min())
            x_normalized = np.zeros_like(x_values) if x_span == 0.0 else (x_values - x_values[0]) / x_span
            y_normalized = np.zeros_like(y_values) if y_span == 0.0 else (y_values - y_values.min()) / y_span

            x_start, y_start = float(x_normalized[0]), float(y_normalized[0])
            x_end, y_end = float(x_normalized[-1]), float(y_normalized[-1])
            denominator = float(np.sqrt((y_end - y_start) ** 2 + (x_end - x_start) ** 2))
            if denominator == 0.0:
                return None

            numerator = np.abs(
                (y_end - y_start) * x_normalized
                - (x_end - x_start) * y_normalized
                + x_end * y_start
                - y_end * x_start
            )
            distances = numerator / denominator
            interior_distances = distances[1:-1]
            if interior_distances.size == 0:
                return None

            elbow_index = int(np.argmax(interior_distances)) + 1
            return int(tested_k_values[elbow_index])
        except Exception as exc:
            self._log(logging.WARNING, f"Fallback usage | elbow detection failed: {exc}")
            return None

    def _fit_selected_k(
        self,
        latent_matrix: np.ndarray,
        selected_k: int,
        config: ClusteringConfig,
    ) -> Tuple[np.ndarray, float, float]:
        try:
            return self._fit_kmeans_and_score(
                latent_matrix=latent_matrix,
                k_value=int(selected_k),
                config=config,
            )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | final clustering fit failed for k={selected_k}: {exc}")
            raise ClusteringStageError("Failed to fit final clustering model.", cause=exc) from exc

    def _fit_kmeans_and_score(
        self,
        latent_matrix: np.ndarray,
        k_value: int,
        config: ClusteringConfig,
    ) -> Tuple[np.ndarray, float, float]:
        try:
            KMeans, silhouette_score = self._load_sklearn_runtime()
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                model = KMeans(
                    n_clusters=int(k_value),
                    random_state=config.random_state,
                    n_init=int(config.n_init),
                    max_iter=int(config.max_iter),
                    algorithm=str(config.algorithm),
                )
                cluster_labels = model.fit_predict(latent_matrix)

            unique_labels = np.unique(cluster_labels)
            if int(unique_labels.size) != int(k_value):
                raise ValueError(
                    f"KMeans produced {unique_labels.size} distinct clusters for requested k={k_value}."
                )
            if int(unique_labels.size) < 2 or int(unique_labels.size) >= int(latent_matrix.shape[0]):
                raise ValueError("Silhouette score requires between 2 and n_samples - 1 distinct clusters.")

            silhouette_kwargs: Dict[str, Any] = {"metric": str(config.silhouette_metric)}
            if config.silhouette_sample_size is not None:
                silhouette_kwargs["sample_size"] = min(int(config.silhouette_sample_size), int(latent_matrix.shape[0]))
                silhouette_kwargs["random_state"] = config.random_state

            silhouette_value = float(silhouette_score(latent_matrix, cluster_labels, **silhouette_kwargs))
            inertia_value = float(model.inertia_)

            for caught_warning in caught_warnings:
                message = str(caught_warning.message)
                if message:
                    self._log(logging.WARNING, f"KMeans warning | k={k_value} | message={message}")

            return np.asarray(cluster_labels, dtype=np.dtype(config.label_dtype)), inertia_value, silhouette_value
        except ClusteringStageError:
            raise
        except Exception as exc:
            raise ValueError(str(exc)) from exc

    def _load_sklearn_runtime(self):
        try:
            if importlib.util.find_spec("sklearn") is None:
                raise ImportError("Missing required dependency 'scikit-learn'. Install it with `pip install scikit-learn`.")

            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            return KMeans, silhouette_score
        except Exception as exc:
            self._log(logging.ERROR, f"Error | scikit-learn runtime unavailable: {exc}")
            raise ClusteringStageError(
                "Clustering stage requires scikit-learn. Install it with `pip install scikit-learn`.",
                cause=exc,
            ) from exc

    def _validate_cluster_labels(
        self,
        cluster_labels: np.ndarray,
        row_count: int,
        best_k: int,
        config: ClusteringConfig,
    ) -> np.ndarray:
        try:
            labels = np.asarray(cluster_labels, dtype=np.dtype(config.label_dtype))
            if labels.ndim != 1:
                raise ValueError("Cluster labels must be a 1D array.")
            if int(labels.shape[0]) != int(row_count):
                raise ValueError(
                    f"Cluster label count mismatch: expected {row_count}, received {labels.shape[0]}."
                )
            unique_labels = np.unique(labels)
            if int(unique_labels.size) < 2:
                raise ValueError("Clustering must produce at least 2 distinct clusters.")
            if int(unique_labels.size) > int(best_k):
                raise ValueError("Distinct cluster label count cannot exceed selected K.")
            return labels
        except Exception as exc:
            self._log(logging.ERROR, f"Error | cluster label validation failed: {exc}")
            raise ClusteringStageError("Invalid clustering labels.", cause=exc) from exc

    def _build_clustering_metadata(
        self,
        report: ClusteringReport,
        cluster_labels: np.ndarray,
        selected_inertia: float,
    ) -> Dict[str, Any]:
        try:
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            return {
                "stage": self.stage_name,
                "row_count": int(report.row_count),
                "input_shape": [int(report.input_shape[0]), int(report.input_shape[1])],
                "tested_k_values": list(report.tested_k_values),
                "silhouette_scores": {int(k): float(v) for k, v in report.silhouette_scores.items()},
                "inertia_values": {int(k): float(v) for k, v in report.inertia_values.items()},
                "selected_k": int(report.selected_k),
                "best_k": int(report.best_k),
                "silhouette_best_k": int(report.silhouette_best_k),
                "elbow_k": None if report.elbow_k is None else int(report.elbow_k),
                "selected_silhouette_score": (
                    None if report.selected_silhouette_score is None else float(report.selected_silhouette_score)
                ),
                "selected_inertia": float(selected_inertia),
                "cluster_count": int(unique_labels.size),
                "cluster_sizes": {
                    int(label): int(count) for label, count in zip(unique_labels.tolist(), counts.tolist())
                },
                "warnings": list(report.warnings),
            }
        except Exception as exc:
            self._log(logging.ERROR, f"Error | clustering metadata construction failed: {exc}")
            raise ClusteringStageError("Failed to build clustering metadata.", cause=exc) from exc

    def _validate_stage_output(self, result: ClusteringResult) -> None:
        try:
            if not isinstance(result, ClusteringResult):
                raise TypeError("Clustering stage output must be a ClusteringResult instance.")
            if result.cluster_labels is None:
                raise ValueError("Clustering stage returned no cluster labels.")
            if int(result.best_k) < 2:
                raise ValueError("best_k must be at least 2.")

            labels = np.asarray(result.cluster_labels)
            metadata = result.clustering_metadata or {}
            if labels.ndim != 1:
                raise ValueError("Cluster labels must be one-dimensional.")
            if int(metadata.get("row_count", labels.shape[0])) != int(labels.shape[0]):
                raise ValueError("Clustering metadata row_count does not match cluster labels.")
            if int(metadata.get("selected_k", result.best_k)) != int(result.best_k):
                raise ValueError("Clustering metadata selected_k does not match best_k.")
            if not np.isfinite(labels.astype(np.float64)).all():
                raise ValueError("Cluster labels contain non-finite values.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage output validation failed: {exc}")
            raise ClusteringStageError("Invalid clustering stage output.", cause=exc) from exc

    def _canonicalize_algorithm(self, algorithm: str) -> str:
        try:
            normalized = str(algorithm).strip().lower().replace("-", "_")
            if normalized in self._algorithm_aliases:
                return self._algorithm_aliases[normalized]
            raise ValueError(f"Unsupported KMeans algorithm '{algorithm}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | algorithm canonicalization failed: {exc}")
            raise ClusteringStageError("Failed to canonicalize clustering algorithm.", cause=exc) from exc

    def _log(self, level: int, message: str) -> None:
        try:
            self.logger.log(level, f"[{self.stage_name}] - {message}")
        except Exception:
            logging.getLogger(__name__).log(level, f"[{self.stage_name}] - {message}")


def cluster_latent_features(
    Z_latent: Optional[Any],
    config: Optional[ClusteringConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> ClusteringResult:
    try:
        module = ClusteringModule(config=config, logger=logger)
        return module.run(Z_latent=Z_latent)
    except ClusteringStageError:
        raise
    except Exception as exc:
        raise ClusteringStageError("Unhandled clustering error.", cause=exc) from exc
