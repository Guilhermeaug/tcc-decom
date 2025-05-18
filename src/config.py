from langchain_huggingface import HuggingFaceEmbeddings
from torch import cuda
from .utils.logger import get_logger

logger = get_logger(__name__)

device = "cuda" if cuda.is_available() else "cpu"
logger.info(f"Using device: {device} for embeddings")

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH = 30
VECTORSTORE_PATH = "embeddings/index"
