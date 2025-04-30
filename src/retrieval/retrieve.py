from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from ..utils.logger import get_logger
from ..config import *
from .config import TEMPLATE

logger = get_logger(__name__)

try:
    logger.info(f"Carregando índice de vetores de: {VECTORSTORE_PATH}")
    db = FAISS.load_local(
        VECTORSTORE_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    logger.info("Índice carregado com sucesso")
except Exception as e:
    logger.error(f"Erro ao carregar índice de vetores: {e}")
    raise

try:
    logger.info("Inicializando modelo de linguagem (LLM)")
    llm = OllamaLLM(model="deepseek-r1:8b")
    logger.info(f"Modelo de linguagem inicializado com sucesso: {llm.model}")
except Exception as e:
    logger.error(f"Erro ao inicializar o modelo de linguagem: {e}")
    raise

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=TEMPLATE,
)

try:
    logger.info("Configurando cadeia de processamento RAG")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    logger.info("Cadeia de processamento RAG configurada com sucesso")
except Exception as e:
    logger.error(f"Erro ao configurar cadeia de processamento RAG: {e}")
    raise


def main():
    logger.info("Iniciando sistema de perguntas e respostas")
    while True:
        question = input("\n❓ Pergunta (ou 'sair'): ")
        if question.lower() == "sair":
            logger.info("Encerrando o sistema")
            break

        query = f"query: {question}"
        logger.info(f"Processando pergunta: {question}")

        logger.debug("Recuperando documentos relevantes")
        retrieved_docs = retriever.invoke(query)
        for i, doc in enumerate(retrieved_docs):
            logger.debug(f"--- Documento {i+1} ---")
            logger.debug(f"Fonte: {doc.metadata.get('source_doc', 'Desconhecido')}")
            logger.debug(f"Conteúdo (primeiros 300 chars): {doc.page_content[:300]}...")

        try:
            logger.info("Gerando resposta...")
            result = qa_chain.invoke({"query": query})

            print("\n🧠 Resposta:")
            print(result["result"])
            logger.info("Resposta gerada com sucesso")

            print("\n📚 Fontes:")
            for doc in result["source_documents"]:
                source = doc.metadata.get("source_doc", "Desconhecido")
                print(f"- {source}")
                logger.debug(f"Utilizando fonte: {source}")
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {e}")
            print(f"\n⚠️ Erro ao processar a pergunta: {e}")


if __name__ == "__main__":
    main()
