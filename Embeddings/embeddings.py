import os

import torch
import gc
import logging
import numpy as np
from typing import List, Dict, Optional, Callable
from sentence_transformers import SentenceTransformer
import time

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Embedding Engine tối ưu hóa cho phần cứng hạn chế (VRAM 4GB).
    Hỗ trợ: Dynamic Batching, Physical VRAM Monitoring, OOM Auto-Recovery.
    """
    
    # Constants cho VRAM thresholds
    VRAM_CRITICAL_THRESHOLD = 350  # MB
    VRAM_WARNING_THRESHOLD = 450   # MB
    VRAM_SAFE_THRESHOLD = 800      # MB

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: Optional[str] = None,
        batch_size: int = 64,
        normalize: bool = True,
        auto_batch: bool = True,
        safety_margin: float = 0.8,
        force_offline: bool = True
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_batch_size = batch_size
        self.current_batch_size = batch_size
        self.normalize = normalize
        self.auto_batch = auto_batch
        self.safety_margin = safety_margin
        self.force_offline = force_offline
        self._calibrated = False
        self._embedding_dim = None  # Cache
        
        logger.info(f"🔧 Initializing EmbeddingEngine: {model_name} on {self.device}")
        logger.info(f"   Batch size: {batch_size} | Auto-batch: {auto_batch}")
        
        # Load model
        self._load_model()
        
        # Get embedding dimension
        self.embedding_dim = self._get_embedding_dimension()
        
        # Warmup GPU if available
        if self.device == "cuda":
            self._warmup_gpu()
        
        logger.info(f"✅ Model loaded. Dimension: {self.embedding_dim} | Device: {self.device}")

    def _load_model(self):
        """Load model với offline fallback"""
        try:
            if self.force_offline:
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                logger.debug("Forcing offline mode")
                
            self.model = SentenceTransformer(
                self.model_name, 
                device=self.device,
                model_kwargs={"local_files_only": self.force_offline}
            )
            
        except Exception as e:
            if self.force_offline:
                logger.warning(f"⚠️ Model not found in cache: {e}")
                logger.info("🔄 Attempting online download...")
                
                # Tạm thời cho phép online
                os.environ.pop('HF_HUB_OFFLINE', None)
                os.environ.pop('TRANSFORMERS_OFFLINE', None)
                
                self.model = SentenceTransformer(self.model_name, device=self.device)
                
                # Khôi phục offline mode
                if self.force_offline:
                    os.environ['HF_HUB_OFFLINE'] = '1'
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'
            else:
                raise e

    def _get_embedding_dimension(self, default: int = 768) -> int:
        """
        Tự động nhận diện embedding dimension từ model
        """
        if self._embedding_dim is not None:
            return self._embedding_dim
        
        model = self.model
        
        # Method 1: SentenceTransformers standard method
        if hasattr(model, "get_sentence_embedding_dimension"):
            try:
                dim = model.get_sentence_embedding_dimension()
                if dim and dim > 0:
                    logger.debug(f"✅ Dimension from SentenceTransformers method: {dim}")
                    self._embedding_dim = int(dim)
                    return self._embedding_dim
            except Exception as e:
                logger.debug(f"Could not get dimension from standard method: {e}")
        
        # Method 2: Model config
        if hasattr(model, "config"):
            config = model.config
            attr_map = {
                "hidden_size": "BERT/RoBERTa/Jina",
                "d_model": "T5/BART",
                "dim": "LLaMA/Mistral",
                "n_embd": "GPT-2",
                "embedding_dim": "CLIP/ViT",
                "num_features": "Vision Models"
            }
            
            for attr, desc in attr_map.items():
                if hasattr(config, attr):
                    val = getattr(config, attr)
                    if isinstance(val, int) and val > 100:
                        logger.debug(f"✅ Dimension from config.{attr} ({desc}): {val}")
                        self._embedding_dim = val
                        return self._embedding_dim
        
        # Method 3: Inspect weights
        try:
            params = list(model.named_parameters())
            
            # 3.1: Find embedding layer
            for name, param in params:
                name_lower = name.lower()
                if ("word_embeddings" in name_lower or "token_embeddings" in name_lower) and len(param.shape) == 2:
                    dim = param.shape[1]
                    logger.debug(f"✅ Dimension from embedding layer ({name}): {dim}")
                    self._embedding_dim = int(dim)
                    return self._embedding_dim
            
            # 3.2: Find last hidden layer
            for name, param in reversed(params):
                shape = param.shape
                if "weight" in name and len(shape) == 2:
                    if shape[1] > 100 and shape[0] <= 100:
                        logger.debug(f"✅ Dimension from projection head: {shape[1]}")
                        self._embedding_dim = int(shape[1])
                        return self._embedding_dim
                    elif shape[1] > 100:
                        logger.debug(f"✅ Dimension from last hidden layer: {shape[1]}")
                        self._embedding_dim = int(shape[1])
                        return self._embedding_dim
        except Exception as e:
            logger.debug(f"Could not inspect weights: {e}")
        
        # Method 4: Fallback - encode dummy text
        try:
            logger.debug("Attempting fallback: encoding dummy text...")
            dummy_emb = self.model.encode(
                ["test"], 
                convert_to_numpy=True,
                show_progress_bar=False
            )[0]
            dim = len(dummy_emb)
            logger.debug(f"✅ Dimension from dummy encoding: {dim}")
            self._embedding_dim = dim
            return self._embedding_dim
        except Exception as e:
            logger.warning(f"Could not determine dimension via dummy encoding: {e}")
        
        # Final fallback
        logger.warning(f"⚠️ Using default dimension: {default}")
        self._embedding_dim = default
        return self._embedding_dim

    def _warmup_gpu(self):
        """Warmup GPU để tránh cold start overhead"""
        try:
            logger.debug("🔥 Warming up GPU...")
            dummy = ["warmup text for GPU initialization"] * 4
            self.model.encode(dummy, batch_size=2, show_progress_bar=False)
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("✅ GPU warmed up")
        except Exception as e:
            logger.debug(f"GPU warmup failed (non-critical): {e}")

    def _get_available_vram_mb(self) -> float:
        """Lấy VRAM còn trống (Physical VRAM)"""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                free_mem, _ = torch.cuda.mem_get_info()
                return free_mem / (1024 ** 2)
            except Exception as e:
                logger.debug(f"Could not get VRAM info: {e}")
                # Trả về inf để trigger conservative behavior
                return float('inf')
        return float('inf')

    def _is_physical_vram_full(self, threshold_mb: int = None) -> bool:
        """Kiểm tra nếu VRAM thực sự sắp hết"""
        if self.device != "cuda": 
            return False
        threshold = threshold_mb or self.VRAM_WARNING_THRESHOLD
        free_vram = self._get_available_vram_mb()
        
        # Nếu free_vram = inf (lỗi đọc), return False
        if free_vram == float('inf'):
            return False
            
        return free_vram < threshold

    def _find_optimal_batch_size(self, sample_texts: List[str] = None):
        """Binary search tìm batch size tối ưu"""
        if not torch.cuda.is_available() or not self.auto_batch:
            logger.debug("Skipping batch size calibration (CPU mode or auto_batch=False)")
            return
        
        logger.info(f"🔍 Calibrating optimal batch size...")
        
        # Check initial VRAM
        initial_vram = self._get_available_vram_mb()
        if initial_vram != float('inf'):
            logger.info(f"📊 Initial free VRAM: {initial_vram:.1f} MB")
            
            if initial_vram < self.VRAM_CRITICAL_THRESHOLD:
                logger.warning("⚠️ VRAM too low for calibration, using conservative batch size")
                self.current_batch_size = max(1, self.initial_batch_size // 4)
                return
        
        # Prepare test samples
        if sample_texts and len(sample_texts) > 0:
            # Lấy top 30-50 samples dài nhất
            sorted_texts = sorted(sample_texts, key=len, reverse=True)
            n_samples = min(50, max(10, len(sorted_texts) // 5))
            test_samples = sorted_texts[:n_samples]
            logger.debug(f"Using {len(test_samples)} longest samples for calibration")
        else:
            # Realistic fallback - tạo text dài
            test_samples = [
                "This is a comprehensive test sentence for batch size calibration in the embedding engine system. " * 20
            ]
            logger.debug("Using generated test samples")
        
        # Binary search
        low, high = 1, self.initial_batch_size
        optimal = 1
        
        while low <= high:
            test_bs = (low + high) // 2
            
            try:
                # Cleanup trước test
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Tạo test batch
                test_batch = []
                while len(test_batch) < test_bs:
                    test_batch.extend(test_samples)
                test_batch = test_batch[:test_bs]
                
                logger.debug(f"Testing batch_size={test_bs}...")
                
                # Test encode
                _ = self.model.encode(
                    test_batch, 
                    batch_size=test_bs, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Kiểm tra VRAM sau test
                if self._is_physical_vram_full(self.VRAM_CRITICAL_THRESHOLD):
                    logger.debug(f"Batch {test_bs} succeeded but VRAM critical")
                    raise RuntimeError("VRAM limit reached")
                
                # Thành công - thử batch lớn hơn
                optimal = test_bs
                low = test_bs + 1
                logger.debug(f"Batch {test_bs} OK, trying larger")
                
            except Exception as e:
                # Thất bại - giảm batch size
                logger.debug(f"Batch {test_bs} failed: {str(e)[:50]}")
                high = test_bs - 1
        
        # Linear probe: thử tăng thêm vài batch từ optimal
        if optimal < self.initial_batch_size:
            for probe_bs in range(optimal + 1, min(optimal + 5, self.initial_batch_size + 1)):
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    test_batch = (test_samples * ((probe_bs // len(test_samples)) + 1))[:probe_bs]
                    _ = self.model.encode(test_batch, batch_size=probe_bs, show_progress_bar=False)
                    
                    if not self._is_physical_vram_full(self.VRAM_CRITICAL_THRESHOLD):
                        optimal = probe_bs
                        logger.debug(f"Linear probe found better batch: {probe_bs}")
                    else:
                        break
                except:
                    break
        
        # Áp dụng safety margin
        self.current_batch_size = max(1, int(optimal * self.safety_margin))
        logger.info(f"✅ Optimal batch size: {self.current_batch_size} (tested up to {optimal}, margin={self.safety_margin})")
        
        # Cleanup
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def _adaptive_encode(
        self, 
        texts: List[str], 
        batch_size: int = None, 
        show_progress: bool = True,
        max_retries: int = 3
    ) -> np.ndarray:
        """Encode với adaptive batching và retry logic"""
        if not texts:
            return np.empty((0, self.embedding_dim))
        
        n_samples = len(texts)
        bs = batch_size or self.current_batch_size
        
        # Xử lý đặc biệt cho số lượng ít
        if n_samples <= 2:
            bs = 1
        
        # Retry loop cho transient errors
        for retry_attempt in range(max_retries):
            current_bs = bs
            
            # Batch size reduction loop
            while current_bs >= 1:
                try:
                    # Pre-encode cleanup nếu VRAM thấp
                    if self.device == "cuda" and self._is_physical_vram_full(self.VRAM_CRITICAL_THRESHOLD):
                        logger.debug("Pre-encode cleanup triggered")
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Encode
                    embeddings = self.model.encode(
                        texts,
                        batch_size=current_bs,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize,
                        show_progress_bar=show_progress and n_samples > 10
                    )
                    
                    # Success - update persistent batch size if reduced
                    if current_bs < self.current_batch_size:
                        self.current_batch_size = current_bs
                        logger.info(f"📉 Batch size adjusted to {current_bs}")
                    
                    return embeddings
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        # OOM - giảm batch size
                        old_bs = current_bs
                        current_bs = max(1, current_bs // 2)
                        
                        logger.warning(f"⚠️ OOM at batch_size={old_bs}, reducing to {current_bs}")
                        
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Nếu đã batch=1 mà vẫn OOM, fallback emergency
                        if old_bs == 1:
                            logger.warning("🚑 Emergency fallback: one-by-one processing")
                            try:
                                embeddings = []
                                for i, text in enumerate(texts):
                                    if i % 50 == 0 and i > 0:
                                        logger.debug(f"Processing {i}/{n_samples}")
                                    
                                    emb = self.model.encode(
                                        [text], 
                                        batch_size=1,
                                        convert_to_numpy=True,
                                        normalize_embeddings=self.normalize,
                                        show_progress_bar=False
                                    )[0]
                                    embeddings.append(emb)
                                    
                                    # Cleanup mỗi 10 samples
                                    if i % 10 == 0 and self.device == "cuda":
                                        torch.cuda.empty_cache()
                                
                                return np.array(embeddings)
                                
                            except Exception as final_err:
                                logger.error(f"💀 Emergency fallback failed: {final_err}")
                                # Trả về zeros thay vì crash
                                return np.zeros((n_samples, self.embedding_dim))
                    else:
                        # Non-OOM error - có thể retry
                        if retry_attempt < max_retries - 1:
                            wait_time = 2 ** retry_attempt
                            logger.warning(f"❌ Error during encoding (attempt {retry_attempt + 1}/{max_retries}): {e}")
                            logger.info(f"Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            break  # Break khỏi batch loop, retry với attempt mới
                        else:
                            logger.error(f"❌ Fatal error after {max_retries} attempts: {e}")
                            raise e
        
        # Nếu thoát retry loop mà vẫn chưa return
        logger.error("💀 All retry attempts exhausted")
        return np.zeros((n_samples, self.embedding_dim))

    def embed_chunks(
        self, 
        chunks: List[Dict], 
        batch_size: int = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Dict]:
        """
        Embed chunks với auto-calibration
        
        Args:
            chunks: List of dicts with 'text' key
            batch_size: Override batch size
            progress_callback: Optional callback(progress: float) for progress tracking
        """
        if not chunks: 
            return []
        
        n_chunks = len(chunks)
        logger.debug(f"📦 Embedding {n_chunks} chunks...")
        
        # Extract texts
        texts = [ch["text"].strip() for ch in chunks]
        
        # Format for model
        formatted = self.format_embedding_input(texts, self.model_name, mode="doc")
        
        # Calibrate batch size if needed
        if self.auto_batch and self.device == "cuda" and not self._calibrated:
            logger.info("🎯 Calibrating batch size with real data...")
            sample_size = min(100, len(formatted))
            self._find_optimal_batch_size(formatted[:sample_size])
            self._calibrated = True
        
        # Encode
        vectors = self._adaptive_encode(formatted, batch_size=batch_size)
        
        # Assign vectors to chunks
        for i, (ch, vec) in enumerate(zip(chunks, vectors)):
            ch["vector"] = vec
            
            # Progress callback
            if progress_callback and i % 50 == 0:
                progress_callback((i + 1) / n_chunks)
        
        logger.debug(f"✅ Embedded {n_chunks} chunks successfully")
        return chunks

    def embed_queries_batch(
        self, 
        queries: List[str], 
        batch_size: int = None
    ) -> np.ndarray:
        """
        Embed queries batch
        
        Args:
            queries: List of query strings
            batch_size: Override batch size
        """
        if not queries: 
            return np.empty((0, self.embedding_dim))
        
        # Calibrate if first time
        if self.auto_batch and self.device == "cuda" and not self._calibrated:
            logger.info("🎯 Calibrating batch size with real queries...")
            formatted_sample = self.format_embedding_input(
                queries[:min(50, len(queries))], 
                self.model_name, 
                mode="query"
            )
            self._find_optimal_batch_size(formatted_sample)
            self._calibrated = True
        
        formatted = self.format_embedding_input(queries, self.model_name, mode="query")
        return self._adaptive_encode(formatted, batch_size=batch_size)

    @staticmethod
    def format_embedding_input(
        texts: List[str], 
        model_name: str, 
        mode: str = "doc"
    ) -> List[str]:
        """
        Format input text theo yêu cầu của model
        
        Args:
            texts: List of input texts
            model_name: Model identifier
            mode: 'doc' or 'query'
        """
        name = model_name.lower()
        
        # E5 models
        if "e5" in name:
            if "instruct" in name:
                if mode == "query":
                    return [f"Instruct: Given a query, retrieve relevant passages\nQuery: {t}" for t in texts]
                else:
                    return [f"passage: {t}" for t in texts]
            else:
                prefix = "query:" if mode == "query" else "passage:"
                return [f"{prefix} {t}" for t in texts]
        
        # Qwen models
        if "qwen" in name:
            prompt = "Given a query, retrieve relevant documents"
            prefix = "Query" if mode == "query" else "Document"
            return [f"Instruct: {prompt}\n{prefix}: {t}" for t in texts]
        
        # BGE, Vietnamese-SBERT, default
        return texts

    def get_vram_report(self) -> Dict:
        """Báo cáo VRAM chi tiết"""
        if self.device != "cuda":
            return {
                "device": "cpu",
                "batch_size": self.current_batch_size,
                "model": self.model_name
            }
        
        try:
            free, total = torch.cuda.mem_get_info()
            free_mb = free / 1024**2
            total_mb = total / 1024**2
            used_mb = total_mb - free_mb
            
            return {
                "device": "cuda",
                "used_mb": round(used_mb, 1),
                "free_mb": round(free_mb, 1),
                "total_mb": round(total_mb, 1),
                "utilization": round(used_mb / total_mb * 100, 1),
                "batch_size": self.current_batch_size,
                "model": self.model_name,
                "calibrated": self._calibrated
            }
        except Exception as e:
            return {
                "device": "cuda",
                "error": str(e),
                "batch_size": self.current_batch_size,
                "model": self.model_name
            }

    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup khi thoát context"""
        logger.debug("Cleaning up EmbeddingEngine...")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return False  # Don't suppress exceptions

    def __repr__(self):
        return f"EmbeddingEngine(model={self.model_name}, device={self.device}, batch={self.current_batch_size})"