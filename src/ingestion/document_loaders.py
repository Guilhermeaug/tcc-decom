import os
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from ..utils.logger import get_logger
from ..config import MIN_CHUNK_LENGTH

logger = get_logger(__name__)


class DocumentLoaderStrategy(ABC):
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        pass


class PDFTextLoader(DocumentLoaderStrategy):
    def load(self, file_path: str) -> List[Document]:
        return PyPDFLoader(file_path).load()


class PDFOCRLoader(DocumentLoaderStrategy):
    def load(self, file_path: str) -> List[Document]:
        return UnstructuredPDFLoader(
            file_path, mode="single", strategy="hi_res", languages=["por"]
        ).load()


class DocxLoader(DocumentLoaderStrategy):
    def load(self, file_path: str) -> List[Document]:
        return Docx2txtLoader(file_path).load()


class DocLoader(DocumentLoaderStrategy):
    def load(self, file_path: str) -> List[Document]:
        return UnstructuredWordDocumentLoader(file_path, mode="single").load()


class ExcelLoader(DocumentLoaderStrategy):
    def load(self, file_path: str) -> List[Document]:
        return UnstructuredExcelLoader(file_path, mode="elements").load()


def load_all_documents(folder_path: str) -> List[Document]:
    all_docs = []
    file_paths = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path):
            file_paths.append(full_path)

    logger.info(
        f"Encontrados {len(file_paths)} arquivos para processamento em {folder_path}"
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        future_to_path = {
            executor.submit(_load_single_document, path): path for path in file_paths
        }

        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                docs_for_file = future.result()
                all_docs.extend(docs_for_file)
            except Exception as exc:
                logger.error(
                    f"Exceção gerada ao carregar {os.path.basename(path)}: {exc}"
                )

    logger.info(f"Total de documentos carregados: {len(all_docs)}")
    return all_docs


def _load_single_document(file_path: str) -> List[Document]:
    try:
        logger.info(f"Iniciando carregamento de: {os.path.basename(file_path)}")
        loader = _choose_loader(file_path)
        docs = loader.load(file_path)
        for doc in docs:
            doc.metadata["source_doc"] = os.path.basename(file_path)
        logger.info(f"Concluído: {os.path.basename(file_path)}")
        return docs
    except Exception as e:
        logger.error(f"Erro ao processar {os.path.basename(file_path)}: {e}")
        return []


def _choose_loader(file_path: str) -> DocumentLoaderStrategy:
    if file_path.endswith(".pdf"):
        try:
            text_loader = PDFTextLoader()
            docs = text_loader.load(file_path)
            is_empty = not docs or all(
                len(doc.page_content.strip()) < MIN_CHUNK_LENGTH for doc in docs
            )
            if docs and not is_empty:
                logger.info(
                    f"Successfully loaded text from: {os.path.basename(file_path)}"
                )
                return text_loader
            else:
                logger.warning(
                    f"No documents returned by text loader. Trying OCR for: {os.path.basename(file_path)}"
                )
                return PDFOCRLoader()
        except Exception as e:
            logger.warning(
                f"Text loading failed ({e}). Trying OCR for: {os.path.basename(file_path)}"
            )
            return PDFOCRLoader()
    elif file_path.endswith(".docx"):
        return DocxLoader()
    elif file_path.endswith(".doc"):
        return DocLoader()
    elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
        return ExcelLoader()
    else:
        error_msg = f"Unsupported file format: {file_path}"
        logger.error(error_msg)
        raise ValueError(f"❌ {error_msg}")
