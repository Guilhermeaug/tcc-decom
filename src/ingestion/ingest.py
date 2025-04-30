from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .document_loaders import load_all_documents
from ..utils.logger import get_logger
from ..config import *

logger = get_logger(__name__)


def main():
    logger.info("Carregando documentos com estratégia...")
    documents = load_all_documents("data/")

    logger.info("Dividindo em chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    logger.info("Filtrando e preparando chunks...")
    filtered_chunks = filter_and_prepare_chunks(chunks)

    logger.info(
        f"Gerando embeddings para {len(filtered_chunks)} chunks e criando vetorstore..."
    )
    db = FAISS.from_documents(filtered_chunks, EMBEDDING_MODEL)
    db.save_local(VECTORSTORE_PATH)

    logger.info("Indexação completa!")


def is_chunk_useful(chunk: Document, min_length: int) -> bool:
    content = chunk.page_content.strip()
    if len(content) < min_length:
        logger.warning(f"Ignorando chunk curto: {content[:100]}...")
        return False
    if content.startswith("http"):
        logger.warning(f"Ignorando chunk de URL: {content[:100]}...")
        return False
    return True


def filter_and_prepare_chunks(
    chunks: List[Document], min_length: int = MIN_CHUNK_LENGTH
) -> List[Document]:
    filtered_chunks = []
    for chunk in chunks:
        if is_chunk_useful(chunk, min_length):
            chunk.page_content = f"passage: {chunk.page_content.strip()}"
            filtered_chunks.append(chunk)
    return filtered_chunks


if __name__ == "__main__":
    main()
