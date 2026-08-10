
import threading

from pathlib import Path
from typing import Dict, Any, Optional

from ndb_host.utils.constants import ModelType, BatchConfig
from ndb_host.db.ndb_settings import NDBConfig
from ndb_host.utils.logger import NebulonDBLogger


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#        Load Configuration
# ==========================================================

cfg = NDBConfig()

# ==========================================================
# Auto batch size helper
# ==========================================================

def get_auto_batch_size(model_type: str = ModelType.EMBEDDING) -> BatchConfig:
    """Decide batch size automatically based on system/device."""

    import torch
    import psutil

    use_cuda = torch.cuda.is_available() and not cfg.NEBULONDB_DEFAULT_MODE
    device = "cuda" if use_cuda else "cpu"

    if device == "cuda":
        free_vram, _ = torch.cuda.mem_get_info()
        free_vram = free_vram / (1024 ** 3)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if model_type == ModelType.EMBEDDING:
            if free_vram >= 20:
                batch = 512
            elif free_vram >= 12:
                batch = 256
            elif free_vram >= 8:
                batch = 128
            elif free_vram >= 4:
                batch = 64
            else:
                batch = 32
        elif model_type == ModelType.CROSS_ENCODER:
            if free_vram >= 20:
                batch = 128
            elif free_vram >= 10:
                batch = 64
            elif free_vram >= 6:
                batch = 32
            else:
                batch = 16
        else:
            batch = 1

        return BatchConfig(
            batch_size=batch,
            device=device,
            dtype=dtype,
            use_fp16=(dtype == torch.float16),
        )
    else:
        ram_gb = psutil.virtual_memory().available / (1024 ** 3)

        if model_type == ModelType.EMBEDDING:
            if ram_gb >= 32:
                batch = 128
            elif ram_gb >= 16:
                batch = 64
            elif ram_gb >= 8:
                batch = 32
            else:
                batch = 16
        elif model_type == ModelType.CROSS_ENCODER:
            if ram_gb >= 32:
                batch = 32
            elif ram_gb >= 16:
                batch = 16
            else:
                batch = 8
        else:
            batch = 8

        return BatchConfig(
            batch_size=batch,
            device="cpu",
            dtype=torch.float32,
            use_fp16=False,
        )

# ==========================================================
# NebulonModelHub – Singleton with class‑level cache
# ==========================================================

class NebulonModelHub:
    _instances: Dict[str, Any] = {}          # model cache: key → loaded model
    _lock = threading.Lock()

    def __new__(cls):
        # No instance state needed; all methods are class‑methods.
        # We keep the class callable for backward compatibility.
        obj = super().__new__(cls)
        return obj

    @classmethod
    def detect_compute(cls):
        import torch
        gpu = torch.cuda.is_available()
        if gpu:
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram > 24:
                return "cuda", torch.bfloat16, "4bit"
            elif vram > 12:
                return "cuda", torch.float16, "4bit"
            else:
                return "cuda", torch.float16, "8bit"
        return "cpu", torch.float32, None

    @classmethod
    def load_model(
        cls,
        model_repo_id: str,
        model_type: str = ModelType.EMBEDDING,
        is_cache_dir: bool = False,
        prefix: Optional[str] = None,
    ):
        """
        Load (and cache) a model. Thread‑safe, single loading per unique repo.

        Args:
            model_repo_id: HuggingFace repo id or local path.
            model_type: ModelType.EMBEDDING or ModelType.CROSS_ENCODER.
            is_cache_dir: if True, resolve the local cache snapshot.
            prefix: organisation prefix (e.g. "sentence-transformers") – only
                    used when is_cache_dir=True to construct the cache path.
        """
        # Resolve local cache snapshot if needed
        if is_cache_dir:
            cache_dir = cfg.NEBULONDB_MODEL_CACHE_DIR
            if prefix:
                cache_folder = Path(cache_dir) / f"models--{prefix}--{model_repo_id}"
            else:
                cache_folder = Path(cache_dir) / f"models--{model_repo_id}"

            refs_path = cache_folder / "refs" / "main"
            if refs_path.exists():
                snapshot_hash = refs_path.read_text().strip()
                model_repo_id = str(cache_folder / "snapshots" / snapshot_hash)
            else:
                # fallback: take the first snapshot
                snapshots = list((cache_folder / "snapshots").iterdir())
                if snapshots:
                    model_repo_id = str(snapshots[0])
                else:
                    raise FileNotFoundError(f"No cached snapshot found for {model_repo_id}")
            logger.info(f"[ModelHub] Loading from cache: {model_repo_id}")

        # Build a cache key
        cache_key = f"{model_type}::{model_repo_id}"

        with cls._lock:
            if cache_key in cls._instances:
                logger.info(f"[ModelHub] Returning cached model for {model_repo_id}")
                return cls._instances[cache_key]

            # Load the model
            logger.info(f"[ModelHub] Loading new model: {model_repo_id}")
            if model_type == ModelType.EMBEDDING:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(
                    str(model_repo_id),
                    cache_folder=str(cfg.NEBULONDB_MODEL_CACHE_DIR),
                    device=cfg.NEBULONDB_EMBEDDING_MODEL_DEVICE,
                )
            elif model_type == ModelType.CROSS_ENCODER:
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(
                    str(model_repo_id),
                    device=cfg.NEBULONDB_CROSS_ENCODER_MODEL_DEVICE,
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            cls._instances[cache_key] = model
            return model

# ==========================================================
# Semantic Embedding Wrapper (uses the singleton hub)
# ==========================================================
class SemanticEmbeddingModel:
    """Wrapper for the embedding model."""

    def __init__(self):
        self.embedding_model_name = cfg.NEBULONDB_EMBEDDING_MODEL
        self.cross_model_name = cfg.NEBULONDB_CROSS_ENCODER_MODEL

    def encode(self, texts, **kwargs):
        model = NebulonModelHub.load_model(self.embedding_model_name)
        return model.encode(texts, **kwargs, batch_size=int(cfg.NEBULONDB_EMBEDDING_BATCH_SIZE))

    def cross_encode(self, texts, **kwargs):
        # Safely split model name into prefix + repo
        if "/" in self.cross_model_name:
            prefix, repo = self.cross_model_name.split("/", 1)
        else:
            prefix, repo = None, self.cross_model_name

        model = NebulonModelHub.load_model(
            model_repo_id=repo,
            model_type=ModelType.CROSS_ENCODER,
            is_cache_dir=True,
            prefix=prefix,
        )
        return model.predict(texts, **kwargs, batch_size=int(cfg.NEBULONDB_CROSS_ENCODER_BATCH_SIZE))

