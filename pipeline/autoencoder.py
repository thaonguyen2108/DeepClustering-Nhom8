from __future__ import annotations

import importlib.util
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AutoencoderConfig:
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    latent_dim: Any = "auto"
    min_latent: int = 8
    max_latent: int = 128
    hidden_dims: Optional[List[int]] = None
    max_hidden_layers: int = 3
    activation: str = "relu"
    loss: str = "mse"
    optimizer: str = "adam"
    use_gpu: bool = True
    device: Optional[str] = None
    output_dtype: str = "float32"
    shuffle: bool = True
    weight_decay: float = 0.0
    num_workers: int = 0
    pin_memory: bool = True
    log_every: int = 1
    random_seed: Optional[int] = 42
    handle_non_finite: str = "replace"
    non_finite_fill_value: float = 0.0
    simple_fallback_enabled: bool = True
    fallback_hidden_dims: Optional[List[int]] = None
    return_model_on_cpu: bool = True


@dataclass
class AutoencoderReport:
    stage: str
    row_count: int
    input_shape: Tuple[int, int]
    feature_sources: List[str] = field(default_factory=list)
    device: Optional[str] = None
    batch_size: int = 128
    epochs: int = 50
    latent_dim: int = 0
    architecture_used: Dict[str, Any] = field(default_factory=dict)
    loss_history: List[float] = field(default_factory=list)
    final_loss: Optional[float] = None
    fallback_used: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class AutoencoderResult:
    Z_latent: np.ndarray
    model: Any
    training_metadata: Dict[str, Any]
    autoencoder_report: AutoencoderReport


class AutoencoderStageError(RuntimeError):
    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        try:
            super().__init__(message)
            self.cause = cause
        except Exception as exc:
            raise RuntimeError("Failed to initialize AutoencoderStageError.") from exc


class AutoencoderModule:
    stage_name = "AUTOENCODER"
    _activation_aliases = {
        "relu": "relu",
        "gelu": "gelu",
        "elu": "elu",
        "leaky_relu": "leaky_relu",
        "leakyrelu": "leaky_relu",
    }
    _loss_aliases = {
        "mse": "mse",
        "mse_loss": "mse",
        "mean_squared_error": "mse",
    }
    _optimizer_aliases = {
        "adam": "adam",
    }
    _non_finite_aliases = {
        "replace": "replace",
        "fill": "replace",
        "raise": "error",
        "error": "error",
    }

    def __init__(
        self,
        config: Optional[AutoencoderConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        try:
            self.config = config or AutoencoderConfig()
            self.logger = logger or logging.getLogger(__name__)
        except Exception as exc:
            raise AutoencoderStageError("Failed to initialize autoencoder module.", cause=exc) from exc

    def run(
        self,
        X_numeric: Optional[Any],
        X_embedding: Optional[Any] = None,
        config: Optional[AutoencoderConfig] = None,
    ) -> AutoencoderResult:
        try:
            active_config = self._validate_config(config or self.config)
            X_final, feature_sources, warnings = self._prepare_training_matrix(
                X_numeric=X_numeric,
                X_embedding=X_embedding,
                config=active_config,
            )
            row_count, input_dim = int(X_final.shape[0]), int(X_final.shape[1])
            self._log(logging.INFO, f"Stage start | rows={row_count} | input_shape={tuple(X_final.shape)}")

            architecture = None
            fallback_used = False
            try:
                architecture = self._select_architecture(input_dim=input_dim, config=active_config)
                self._log(
                    logging.INFO,
                    f"Architecture selected | encoder={[input_dim] + list(architecture['hidden_dims']) + [architecture['latent_dim']]}",
                )
                self._log(logging.INFO, f"Latent dimension | value={architecture['latent_dim']}")
                model, loss_history, device_name = self._train_model(
                    X_final=X_final,
                    architecture=architecture,
                    config=active_config,
                )
            except Exception as primary_exc:
                if not active_config.simple_fallback_enabled or not self._is_recoverable_fallback_error(primary_exc):
                    raise
                fallback_used = True
                warnings.append(
                    f"Primary autoencoder architecture failed. Falling back to simpler architecture: {primary_exc}"
                )
                self._log(
                    logging.WARNING,
                    f"Fallback usage | mode=simpler_architecture | reason={type(primary_exc).__name__}: {primary_exc}",
                )
                architecture = self._build_fallback_architecture(input_dim=input_dim, config=active_config)
                self._log(
                    logging.INFO,
                    f"Architecture selected | encoder={[input_dim] + list(architecture['hidden_dims']) + [architecture['latent_dim']]}",
                )
                self._log(logging.INFO, f"Latent dimension | value={architecture['latent_dim']}")
                model, loss_history, device_name = self._train_model(
                    X_final=X_final,
                    architecture=architecture,
                    config=active_config,
                )

            final_loss = float(loss_history[-1]) if loss_history else None
            Z_latent = self._encode_latent(
                X_final=X_final,
                model=model,
                architecture=architecture,
                config=active_config,
                device_name=device_name,
                warnings=warnings,
            )

            if active_config.return_model_on_cpu and hasattr(model, "to"):
                model = model.to("cpu")

            if final_loss is not None:
                self._log(logging.INFO, f"Final loss | value={final_loss:.6f}")

            report = AutoencoderReport(
                stage=self.stage_name,
                row_count=row_count,
                input_shape=(row_count, input_dim),
                feature_sources=feature_sources,
                device=device_name,
                batch_size=int(active_config.batch_size),
                epochs=int(active_config.epochs),
                latent_dim=int(architecture["latent_dim"]),
                architecture_used=dict(architecture),
                loss_history=[float(value) for value in loss_history],
                final_loss=final_loss,
                fallback_used=fallback_used,
                warnings=warnings,
            )
            training_metadata = self._build_training_metadata(
                report=report,
                latent_shape=tuple(Z_latent.shape),
            )
            result = AutoencoderResult(
                Z_latent=Z_latent,
                model=model,
                training_metadata=training_metadata,
                autoencoder_report=report,
            )
            self._validate_stage_output(result)
            return result
        except AutoencoderStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage failed with {type(exc).__name__}: {exc}")
            raise AutoencoderStageError("Autoencoder stage failed.", cause=exc) from exc

    def _is_recoverable_fallback_error(self, exc: Exception) -> bool:
        try:
            current: Optional[BaseException] = exc
            depth = 0
            collected_messages: List[str] = []

            while current is not None and depth < 8:
                collected_messages.append(str(current))
                if isinstance(current, ModuleNotFoundError):
                    return False
                nested_cause = getattr(current, "cause", None)
                current = nested_cause or getattr(current, "__cause__", None)
                depth += 1

            combined_message = " ".join(collected_messages).lower()
            critical_markers = (
                "pytorch runtime is unavailable",
                "autoencoder stage requires pytorch",
                "no module named 'torch'",
                "failed to prepare autoencoder input matrix",
                "failed to coerce",
                "failed to sanitize autoencoder features",
            )
            return not any(marker in combined_message for marker in critical_markers)
        except Exception:
            return False

    def _validate_config(self, config: AutoencoderConfig) -> AutoencoderConfig:
        try:
            normalized = AutoencoderConfig(
                epochs=max(1, int(config.epochs)),
                batch_size=max(1, int(config.batch_size)),
                learning_rate=float(config.learning_rate),
                latent_dim=config.latent_dim,
                min_latent=max(1, int(config.min_latent)),
                max_latent=max(1, int(config.max_latent)),
                hidden_dims=list(config.hidden_dims) if config.hidden_dims is not None else None,
                max_hidden_layers=max(0, int(config.max_hidden_layers)),
                activation=self._canonicalize_activation(config.activation),
                loss=self._canonicalize_loss(config.loss),
                optimizer=self._canonicalize_optimizer(config.optimizer),
                use_gpu=bool(config.use_gpu),
                device=str(config.device) if config.device is not None else None,
                output_dtype=str(np.dtype(config.output_dtype)),
                shuffle=bool(config.shuffle),
                weight_decay=max(0.0, float(config.weight_decay)),
                num_workers=max(0, int(config.num_workers)),
                pin_memory=bool(config.pin_memory),
                log_every=max(1, int(config.log_every)),
                random_seed=config.random_seed if config.random_seed is None else int(config.random_seed),
                handle_non_finite=self._canonicalize_non_finite_mode(config.handle_non_finite),
                non_finite_fill_value=float(config.non_finite_fill_value),
                simple_fallback_enabled=bool(config.simple_fallback_enabled),
                fallback_hidden_dims=list(config.fallback_hidden_dims)
                if config.fallback_hidden_dims is not None
                else None,
                return_model_on_cpu=bool(config.return_model_on_cpu),
            )
            if normalized.learning_rate <= 0:
                raise ValueError("learning_rate must be greater than 0.")
            if normalized.max_latent < normalized.min_latent:
                normalized.max_latent = normalized.min_latent
            return normalized
        except Exception as exc:
            self._log(logging.ERROR, f"Error | invalid autoencoder config: {exc}")
            raise AutoencoderStageError("Invalid autoencoder configuration.", cause=exc) from exc

    def _prepare_training_matrix(
        self,
        X_numeric: Optional[Any],
        X_embedding: Optional[Any],
        config: AutoencoderConfig,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        try:
            warnings: List[str] = []
            numeric_matrix = self._coerce_matrix(
                matrix=X_numeric,
                name="X_numeric",
                config=config,
                warnings=warnings,
            )
            embedding_matrix = self._coerce_matrix(
                matrix=X_embedding,
                name="X_embedding",
                config=config,
                warnings=warnings,
            )

            feature_sources: List[str] = []
            active_matrices: List[np.ndarray] = []
            row_count: Optional[int] = None

            for source_name, matrix in (("numeric", numeric_matrix), ("embedding", embedding_matrix)):
                if matrix is None or int(matrix.shape[1]) == 0:
                    continue
                if row_count is None:
                    row_count = int(matrix.shape[0])
                elif int(matrix.shape[0]) != row_count:
                    raise ValueError(
                        f"Feature source row mismatch for '{source_name}': expected {row_count}, received {matrix.shape[0]}."
                    )
                feature_sources.append(source_name)
                active_matrices.append(matrix)

            if not active_matrices:
                raise ValueError("Autoencoder requires at least one non-empty feature source.")

            X_final = active_matrices[0] if len(active_matrices) == 1 else np.concatenate(active_matrices, axis=1)
            X_final = self._sanitize_array(
                array=X_final,
                scope="X_final",
                config=config,
                warnings=warnings,
            )

            if X_final.ndim != 2:
                raise ValueError("Merged autoencoder input must be a 2D matrix.")
            if int(X_final.shape[0]) <= 0:
                raise ValueError("Merged autoencoder input does not contain any rows.")
            if int(X_final.shape[1]) <= 0:
                raise ValueError("Merged autoencoder input does not contain any features.")

            return X_final.astype(np.dtype(config.output_dtype), copy=False), feature_sources, warnings
        except Exception as exc:
            self._log(logging.ERROR, f"Error | feature merge failed: {exc}")
            raise AutoencoderStageError("Failed to prepare autoencoder input matrix.", cause=exc) from exc

    def _coerce_matrix(
        self,
        matrix: Optional[Any],
        name: str,
        config: AutoencoderConfig,
        warnings: List[str],
    ) -> Optional[np.ndarray]:
        try:
            if matrix is None:
                return None

            if hasattr(matrix, "to_numpy"):
                array = np.asarray(matrix.to_numpy(), dtype=np.dtype(config.output_dtype))
            else:
                array = np.asarray(matrix, dtype=np.dtype(config.output_dtype))

            if array.ndim == 0:
                raise ValueError(f"{name} must be at least 1-dimensional.")
            if array.ndim == 1:
                if array.size == 0:
                    array = array.reshape(0, 0)
                else:
                    array = array.reshape(-1, 1)
            if array.ndim != 2:
                raise ValueError(f"{name} must be a 2D numeric matrix.")

            return self._sanitize_array(
                array=array,
                scope=name,
                config=config,
                warnings=warnings,
            )
        except Exception as exc:
            self._log(logging.ERROR, f"Error | failed to coerce {name}: {exc}")
            raise AutoencoderStageError(f"Failed to coerce {name} into a numeric matrix.", cause=exc) from exc

    def _sanitize_array(
        self,
        array: np.ndarray,
        scope: str,
        config: AutoencoderConfig,
        warnings: List[str],
    ) -> np.ndarray:
        try:
            cast_array = np.asarray(array, dtype=np.dtype(config.output_dtype))
            if cast_array.size == 0 or np.isfinite(cast_array).all():
                return cast_array

            if config.handle_non_finite == "replace":
                warnings.append(f"{scope} contained non-finite values and they were replaced.")
                self._log(logging.WARNING, f"Fallback usage | scope={scope} | action=replace_non_finite")
                return np.nan_to_num(
                    cast_array,
                    nan=config.non_finite_fill_value,
                    posinf=config.non_finite_fill_value,
                    neginf=config.non_finite_fill_value,
                ).astype(np.dtype(config.output_dtype), copy=False)

            raise ValueError(f"{scope} contains non-finite values.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | non-finite handling failed for {scope}: {exc}")
            raise AutoencoderStageError("Failed to sanitize autoencoder features.", cause=exc) from exc

    def _select_architecture(
        self,
        input_dim: int,
        config: AutoencoderConfig,
    ) -> Dict[str, Any]:
        try:
            latent_dim = self._resolve_latent_dim(input_dim=input_dim, config=config)
            hidden_dims = self._resolve_hidden_dims(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=config.hidden_dims,
                max_hidden_layers=config.max_hidden_layers,
            )
            architecture = {
                "strategy": "configured" if config.hidden_dims is not None else "auto",
                "input_dim": int(input_dim),
                "hidden_dims": hidden_dims,
                "latent_dim": int(latent_dim),
                "decoder_hidden_dims": list(reversed(hidden_dims)),
                "activation": config.activation,
            }
            return architecture
        except Exception as exc:
            self._log(logging.ERROR, f"Error | architecture selection failed: {exc}")
            raise AutoencoderStageError("Failed to select autoencoder architecture.", cause=exc) from exc

    def _build_fallback_architecture(
        self,
        input_dim: int,
        config: AutoencoderConfig,
    ) -> Dict[str, Any]:
        try:
            try:
                latent_dim = self._resolve_latent_dim(input_dim=input_dim, config=config)
            except Exception:
                latent_dim = self._resolve_auto_latent_dim(
                    input_dim=input_dim,
                    min_latent=config.min_latent,
                    max_latent=config.max_latent,
                )

            hidden_dims = self._resolve_fallback_hidden_dims(
                input_dim=input_dim,
                latent_dim=latent_dim,
                fallback_hidden_dims=config.fallback_hidden_dims,
            )
            return {
                "strategy": "fallback_simple",
                "input_dim": int(input_dim),
                "hidden_dims": hidden_dims,
                "latent_dim": int(latent_dim),
                "decoder_hidden_dims": list(reversed(hidden_dims)),
                "activation": config.activation,
            }
        except Exception as exc:
            self._log(logging.ERROR, f"Error | fallback architecture selection failed: {exc}")
            raise AutoencoderStageError("Failed to build fallback autoencoder architecture.", cause=exc) from exc

    def _resolve_latent_dim(
        self,
        input_dim: int,
        config: AutoencoderConfig,
    ) -> int:
        try:
            if input_dim <= 0:
                raise ValueError("input_dim must be greater than 0.")

            raw_latent = config.latent_dim
            if isinstance(raw_latent, str):
                normalized = raw_latent.strip().lower()
                if normalized != "auto":
                    raise ValueError(f"Unsupported latent_dim value '{raw_latent}'.")
                return self._resolve_auto_latent_dim(
                    input_dim=input_dim,
                    min_latent=config.min_latent,
                    max_latent=config.max_latent,
                )

            requested = int(raw_latent)
            if requested <= 0:
                raise ValueError("latent_dim must be greater than 0.")

            max_compact_latent = input_dim if input_dim <= 2 else max(1, input_dim // 2)
            effective_latent = min(requested, max_compact_latent, config.max_latent)
            return max(1, effective_latent)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | latent dimension resolution failed: {exc}")
            raise AutoencoderStageError("Failed to resolve latent dimension.", cause=exc) from exc

    def _resolve_auto_latent_dim(
        self,
        input_dim: int,
        min_latent: int,
        max_latent: int,
    ) -> int:
        try:
            if input_dim <= 1:
                return 1

            max_compact_latent = max(1, min(max_latent, input_dim // 2))
            min_effective_latent = min(min_latent, max_compact_latent)
            candidate = int(round(math.sqrt(float(input_dim)) * 2.0))
            candidate = max(min_effective_latent, candidate)
            candidate = min(candidate, max_compact_latent)
            return max(1, candidate)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | auto latent dimension resolution failed: {exc}")
            raise AutoencoderStageError("Failed to resolve automatic latent dimension.", cause=exc) from exc

    def _resolve_hidden_dims(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[Sequence[int]],
        max_hidden_layers: int,
    ) -> List[int]:
        try:
            if hidden_dims is not None:
                validated = [int(value) for value in hidden_dims]
                return self._validate_hidden_dims(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    hidden_dims=validated,
                )

            if input_dim <= latent_dim + 1 or max_hidden_layers <= 0:
                return []

            ratio = max(1.0, float(input_dim) / float(max(1, latent_dim)))
            layer_count = min(max_hidden_layers, max(1, int(math.log2(ratio))))
            geometric_steps = np.geomspace(input_dim, max(1, latent_dim), num=layer_count + 2)[1:-1]

            auto_hidden_dims: List[int] = []
            previous_dim = int(input_dim)
            for raw_value in geometric_steps:
                hidden_dim = int(round(float(raw_value)))
                hidden_dim = min(hidden_dim, previous_dim - 1)
                if hidden_dim <= latent_dim:
                    continue
                auto_hidden_dims.append(hidden_dim)
                previous_dim = hidden_dim

            return self._deduplicate_hidden_dims(auto_hidden_dims, latent_dim=latent_dim)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | hidden dimension resolution failed: {exc}")
            raise AutoencoderStageError("Failed to resolve hidden dimensions.", cause=exc) from exc

    def _resolve_fallback_hidden_dims(
        self,
        input_dim: int,
        latent_dim: int,
        fallback_hidden_dims: Optional[Sequence[int]],
    ) -> List[int]:
        try:
            if fallback_hidden_dims is not None:
                return self._validate_hidden_dims(
                    input_dim=input_dim,
                    latent_dim=latent_dim,
                    hidden_dims=[int(value) for value in fallback_hidden_dims],
                )

            if input_dim <= latent_dim + 1:
                return []

            bridge_dim = min(input_dim - 1, max(latent_dim + 1, int(round((input_dim + latent_dim) / 2.0))))
            if bridge_dim <= latent_dim:
                return []
            return [int(bridge_dim)]
        except Exception as exc:
            self._log(logging.ERROR, f"Error | fallback hidden dimension resolution failed: {exc}")
            raise AutoencoderStageError("Failed to resolve fallback hidden dimensions.", cause=exc) from exc

    def _validate_hidden_dims(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
    ) -> List[int]:
        try:
            validated: List[int] = []
            previous_dim = int(input_dim)
            for hidden_dim in hidden_dims:
                current_dim = int(hidden_dim)
                if current_dim <= latent_dim:
                    raise ValueError("Hidden dimensions must be greater than latent_dim.")
                if current_dim >= previous_dim:
                    raise ValueError("Hidden dimensions must decrease strictly from input to latent space.")
                validated.append(current_dim)
                previous_dim = current_dim
            return self._deduplicate_hidden_dims(validated, latent_dim=latent_dim)
        except Exception as exc:
            self._log(logging.ERROR, f"Error | hidden dimension validation failed: {exc}")
            raise AutoencoderStageError("Failed to validate hidden dimensions.", cause=exc) from exc

    def _deduplicate_hidden_dims(
        self,
        hidden_dims: Sequence[int],
        latent_dim: int,
    ) -> List[int]:
        try:
            deduplicated: List[int] = []
            previous_dim: Optional[int] = None
            for hidden_dim in hidden_dims:
                if hidden_dim <= latent_dim:
                    continue
                if previous_dim is not None and hidden_dim >= previous_dim:
                    continue
                deduplicated.append(int(hidden_dim))
                previous_dim = int(hidden_dim)
            return deduplicated
        except Exception as exc:
            self._log(logging.ERROR, f"Error | hidden dimension deduplication failed: {exc}")
            raise AutoencoderStageError("Failed to clean hidden dimension list.", cause=exc) from exc

    def _train_model(
        self,
        X_final: np.ndarray,
        architecture: Dict[str, Any],
        config: AutoencoderConfig,
    ) -> Tuple[Any, List[float], str]:
        try:
            torch, nn, DataLoader, TensorDataset = self._load_torch_runtime()
            if config.random_seed is not None:
                self._set_random_seed(torch=torch, seed=int(config.random_seed))

            device = self._resolve_device(torch=torch, config=config)
            model = self._build_model(nn=nn, architecture=architecture).to(device)
            criterion = self._build_loss(nn=nn, config=config)
            optimizer = self._build_optimizer(torch=torch, model=model, config=config)

            batch_size = min(int(config.batch_size), int(X_final.shape[0]))
            pin_memory = bool(config.pin_memory and str(device).startswith("cuda"))
            self._log(
                logging.INFO,
                f"Training setup | device={device} | batch_size={batch_size} | epochs={config.epochs}",
            )

            train_tensor = torch.tensor(X_final, dtype=torch.float32)
            train_dataset = TensorDataset(train_tensor)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=bool(config.shuffle),
                num_workers=int(config.num_workers),
                pin_memory=pin_memory,
            )

            loss_history: List[float] = []
            for epoch_index in range(1, int(config.epochs) + 1):
                model.train()
                total_loss = 0.0
                total_samples = 0

                for (batch_inputs,) in train_loader:
                    batch_inputs = batch_inputs.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    reconstructed, _ = model(batch_inputs)
                    loss = criterion(reconstructed, batch_inputs)
                    if not bool(torch.isfinite(loss).all().item()):
                        raise RuntimeError("Encountered non-finite training loss.")
                    loss.backward()
                    optimizer.step()

                    batch_size_value = int(batch_inputs.shape[0])
                    total_loss += float(loss.detach().item()) * batch_size_value
                    total_samples += batch_size_value

                epoch_loss = total_loss / max(1, total_samples)
                loss_history.append(float(epoch_loss))
                if epoch_index == 1 or epoch_index == int(config.epochs) or epoch_index % int(config.log_every) == 0:
                    self._log(
                        logging.INFO,
                        f"Training progress | epoch={epoch_index}/{config.epochs} | loss={epoch_loss:.6f}",
                    )

            if not loss_history:
                raise ValueError("Autoencoder training did not produce any loss values.")
            return model, loss_history, str(device)
        except AutoencoderStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | model training failed: {exc}")
            raise AutoencoderStageError("Failed to train autoencoder model.", cause=exc) from exc

    def _encode_latent(
        self,
        X_final: np.ndarray,
        model: Any,
        architecture: Dict[str, Any],
        config: AutoencoderConfig,
        device_name: str,
        warnings: List[str],
    ) -> np.ndarray:
        try:
            torch, _, DataLoader, TensorDataset = self._load_torch_runtime()
            device = torch.device(device_name)
            batch_size = min(int(config.batch_size), int(X_final.shape[0]))
            pin_memory = bool(config.pin_memory and str(device).startswith("cuda"))

            latent_tensor = torch.tensor(X_final, dtype=torch.float32)
            latent_dataset = TensorDataset(latent_tensor)
            latent_loader = DataLoader(
                latent_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(config.num_workers),
                pin_memory=pin_memory,
            )

            model.eval()
            latent_batches: List[np.ndarray] = []
            with torch.no_grad():
                for (batch_inputs,) in latent_loader:
                    batch_inputs = batch_inputs.to(device)
                    latent_batch = model.encode(batch_inputs)
                    latent_batches.append(latent_batch.detach().cpu().numpy())

            if latent_batches:
                Z_latent = np.concatenate(latent_batches, axis=0)
            else:
                Z_latent = np.empty((0, int(architecture["latent_dim"])), dtype=np.dtype(config.output_dtype))

            Z_latent = self._sanitize_array(
                array=Z_latent,
                scope="Z_latent",
                config=config,
                warnings=warnings,
            )
            if Z_latent.ndim != 2:
                raise ValueError("Latent representation must be a 2D matrix.")
            if int(Z_latent.shape[0]) != int(X_final.shape[0]):
                raise ValueError(
                    f"Latent row count mismatch: expected {X_final.shape[0]}, received {Z_latent.shape[0]}."
                )
            if int(Z_latent.shape[1]) != int(architecture["latent_dim"]):
                raise ValueError(
                    f"Latent column count mismatch: expected {architecture['latent_dim']}, received {Z_latent.shape[1]}."
                )

            return Z_latent.astype(np.dtype(config.output_dtype), copy=False)
        except AutoencoderStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | latent extraction failed: {exc}")
            raise AutoencoderStageError("Failed to extract latent representation.", cause=exc) from exc

    def _load_torch_runtime(self):
        try:
            self._ensure_torch_dependency()
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset

            return torch, nn, DataLoader, TensorDataset
        except AutoencoderStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | torch runtime unavailable: {exc}")
            raise AutoencoderStageError(
                "PyTorch runtime is unavailable for autoencoder training. Install it with `pip install torch`.",
                cause=exc,
            ) from exc

    def _resolve_device(
        self,
        torch: Any,
        config: AutoencoderConfig,
    ) -> Any:
        try:
            if config.device is not None:
                return torch.device(config.device)
            if config.use_gpu and torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | device resolution failed: {exc}")
            raise AutoencoderStageError("Failed to resolve autoencoder device.", cause=exc) from exc

    def _set_random_seed(
        self,
        torch: Any,
        seed: int,
    ) -> None:
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception as exc:
            self._log(logging.WARNING, f"Fallback usage | random seed initialization failed: {exc}")

    def _build_model(
        self,
        nn: Any,
        architecture: Dict[str, Any],
    ) -> Any:
        try:
            input_dim = int(architecture["input_dim"])
            hidden_dims = [int(value) for value in architecture["hidden_dims"]]
            latent_dim = int(architecture["latent_dim"])
            activation_factory = self._build_activation_factory(nn=nn, activation_name=architecture["activation"])

            class TabularAutoencoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    encoder_layers: List[Any] = []
                    previous_dim = input_dim
                    for hidden_dim in hidden_dims:
                        encoder_layers.append(nn.Linear(previous_dim, hidden_dim))
                        encoder_layers.append(activation_factory())
                        previous_dim = hidden_dim
                    encoder_layers.append(nn.Linear(previous_dim, latent_dim))
                    self.encoder = nn.Sequential(*encoder_layers)

                    decoder_layers: List[Any] = []
                    previous_dim = latent_dim
                    for hidden_dim in reversed(hidden_dims):
                        decoder_layers.append(nn.Linear(previous_dim, hidden_dim))
                        decoder_layers.append(activation_factory())
                        previous_dim = hidden_dim
                    decoder_layers.append(nn.Linear(previous_dim, input_dim))
                    self.decoder = nn.Sequential(*decoder_layers)

                def encode(self, inputs):
                    return self.encoder(inputs)

                def decode(self, latent):
                    return self.decoder(latent)

                def forward(self, inputs):
                    latent = self.encode(inputs)
                    reconstructed = self.decode(latent)
                    return reconstructed, latent

            model = TabularAutoencoder()
            model.architecture = dict(architecture)
            return model
        except Exception as exc:
            self._log(logging.ERROR, f"Error | model construction failed: {exc}")
            raise AutoencoderStageError("Failed to construct autoencoder model.", cause=exc) from exc

    def _build_activation_factory(
        self,
        nn: Any,
        activation_name: str,
    ):
        try:
            if activation_name == "relu":
                return lambda: nn.ReLU()
            if activation_name == "gelu":
                return lambda: nn.GELU()
            if activation_name == "elu":
                return lambda: nn.ELU()
            if activation_name == "leaky_relu":
                return lambda: nn.LeakyReLU()
            raise ValueError(f"Unsupported activation '{activation_name}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | activation factory construction failed: {exc}")
            raise AutoencoderStageError("Failed to construct activation factory.", cause=exc) from exc

    def _build_loss(
        self,
        nn: Any,
        config: AutoencoderConfig,
    ) -> Any:
        try:
            if config.loss == "mse":
                return nn.MSELoss()
            raise ValueError(f"Unsupported loss '{config.loss}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | loss construction failed: {exc}")
            raise AutoencoderStageError("Failed to construct training loss.", cause=exc) from exc

    def _build_optimizer(
        self,
        torch: Any,
        model: Any,
        config: AutoencoderConfig,
    ) -> Any:
        try:
            if config.optimizer == "adam":
                return torch.optim.Adam(
                    model.parameters(),
                    lr=float(config.learning_rate),
                    weight_decay=float(config.weight_decay),
                )
            raise ValueError(f"Unsupported optimizer '{config.optimizer}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | optimizer construction failed: {exc}")
            raise AutoencoderStageError("Failed to construct optimizer.", cause=exc) from exc

    def _build_training_metadata(
        self,
        report: AutoencoderReport,
        latent_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        try:
            return {
                "stage": self.stage_name,
                "row_count": int(report.row_count),
                "input_shape": list(report.input_shape),
                "latent_shape": [int(latent_shape[0]), int(latent_shape[1])],
                "feature_sources": list(report.feature_sources),
                "device": report.device,
                "batch_size": int(report.batch_size),
                "epochs": int(report.epochs),
                "latent_dim": int(report.latent_dim),
                "architecture_used": dict(report.architecture_used),
                "loss_history": [float(value) for value in report.loss_history],
                "final_loss": None if report.final_loss is None else float(report.final_loss),
                "fallback_used": bool(report.fallback_used),
                "warnings": list(report.warnings),
            }
        except Exception as exc:
            self._log(logging.ERROR, f"Error | training metadata construction failed: {exc}")
            raise AutoencoderStageError("Failed to build autoencoder training metadata.", cause=exc) from exc

    def _ensure_torch_dependency(self) -> None:
        try:
            if importlib.util.find_spec("torch") is None:
                raise AutoencoderStageError(
                    "Autoencoder stage requires PyTorch. Install it with `pip install torch`."
                )
        except AutoencoderStageError:
            raise
        except Exception as exc:
            self._log(logging.ERROR, f"Error | torch dependency check failed: {exc}")
            raise AutoencoderStageError("PyTorch dependency is unavailable for the autoencoder stage.", cause=exc) from exc

    def _validate_stage_output(self, result: AutoencoderResult) -> None:
        try:
            if not isinstance(result, AutoencoderResult):
                raise TypeError("Autoencoder stage output must be an AutoencoderResult instance.")
            if result.model is None:
                raise ValueError("Autoencoder stage did not return a trained model.")

            Z_latent = np.asarray(result.Z_latent)
            metadata = result.training_metadata or {}
            report = result.autoencoder_report

            if Z_latent.ndim != 2:
                raise ValueError("Latent representation must be a 2D matrix.")
            if Z_latent.size == 0:
                raise ValueError("Latent representation cannot be empty.")
            if not np.isfinite(Z_latent).all():
                raise ValueError("Latent representation contains non-finite values.")
            if int(metadata.get("row_count", Z_latent.shape[0])) != int(Z_latent.shape[0]):
                raise ValueError("Training metadata row_count does not match latent output.")
            if int(metadata.get("latent_dim", Z_latent.shape[1])) != int(Z_latent.shape[1]):
                raise ValueError("Training metadata latent_dim does not match latent output.")
            if list(metadata.get("latent_shape", [])) != [int(Z_latent.shape[0]), int(Z_latent.shape[1])]:
                raise ValueError("Training metadata latent_shape does not match latent output.")
            if not report.loss_history:
                raise ValueError("Autoencoder report does not contain training loss history.")
            if report.final_loss is None or not np.isfinite(float(report.final_loss)):
                raise ValueError("Autoencoder final loss must be a finite value.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | stage output validation failed: {exc}")
            raise AutoencoderStageError("Invalid autoencoder stage output.", cause=exc) from exc

    def _canonicalize_activation(self, activation: str) -> str:
        try:
            normalized = str(activation).strip().lower().replace("-", "_")
            if normalized in self._activation_aliases:
                return self._activation_aliases[normalized]
            raise ValueError(f"Unsupported activation '{activation}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | activation canonicalization failed: {exc}")
            raise AutoencoderStageError("Failed to canonicalize activation.", cause=exc) from exc

    def _canonicalize_loss(self, loss: str) -> str:
        try:
            normalized = str(loss).strip().lower().replace("-", "_")
            if normalized in self._loss_aliases:
                return self._loss_aliases[normalized]
            raise ValueError(f"Unsupported loss '{loss}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | loss canonicalization failed: {exc}")
            raise AutoencoderStageError("Failed to canonicalize loss.", cause=exc) from exc

    def _canonicalize_optimizer(self, optimizer: str) -> str:
        try:
            normalized = str(optimizer).strip().lower().replace("-", "_")
            if normalized in self._optimizer_aliases:
                return self._optimizer_aliases[normalized]
            raise ValueError(f"Unsupported optimizer '{optimizer}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | optimizer canonicalization failed: {exc}")
            raise AutoencoderStageError("Failed to canonicalize optimizer.", cause=exc) from exc

    def _canonicalize_non_finite_mode(self, mode: str) -> str:
        try:
            normalized = str(mode).strip().lower().replace("-", "_")
            if normalized in self._non_finite_aliases:
                return self._non_finite_aliases[normalized]
            raise ValueError(f"Unsupported non-finite handling mode '{mode}'.")
        except Exception as exc:
            self._log(logging.ERROR, f"Error | non-finite mode canonicalization failed: {exc}")
            raise AutoencoderStageError("Failed to canonicalize non-finite handling mode.", cause=exc) from exc

    def _log(self, level: int, message: str) -> None:
        try:
            self.logger.log(level, f"[{self.stage_name}] - {message}")
        except Exception:
            logging.getLogger(__name__).log(level, f"[{self.stage_name}] - {message}")


def train_autoencoder_features(
    X_numeric: Optional[Any],
    X_embedding: Optional[Any] = None,
    config: Optional[AutoencoderConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> AutoencoderResult:
    try:
        module = AutoencoderModule(config=config, logger=logger)
        return module.run(X_numeric=X_numeric, X_embedding=X_embedding)
    except AutoencoderStageError:
        raise
    except Exception as exc:
        raise AutoencoderStageError("Unhandled autoencoder error.", cause=exc) from exc
