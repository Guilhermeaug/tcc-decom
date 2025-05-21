from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from ..utils.logger import get_logger
from ..config import *
from .config import TEMPLATE
from .llm_config import get_llm, LLMProvider

load_dotenv()

logger = get_logger(__name__)

try:
    logger.info(f"Carregando índice de vetores de: {VECTORSTORE_PATH}")
    db = FAISS.load_local(
        VECTORSTORE_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    logger.info("Índice carregado com sucesso")
except Exception as e:
    logger.error(f"Erro ao carregar índice de vetores: {e}")
    raise


def initialize_llm(provider: LLMProvider = "openai", model="gpt-4o-mini", **kwargs):
    try:
        logger.info(f"Inicializando modelo de linguagem (LLM) - Provider: {provider}")
        llm = get_llm(provider=provider, model=model, **kwargs)
        logger.info(f"Modelo de linguagem inicializado com sucesso: {model}")
        return llm
    except Exception as e:
        logger.error(f"Erro ao inicializar o modelo de linguagem: {e}")
        raise


def create_qa_chain(llm, retriever):
    try:
        logger.info("Configurando cadeia de processamento RAG")

        document_chain_prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=TEMPLATE,
        )
        combine_docs_chain = create_stuff_documents_chain(llm, document_chain_prompt)
        qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

        logger.info("Cadeia de processamento RAG configurada com sucesso")
        return qa_chain
    except Exception as e:
        logger.error(f"Erro ao configurar cadeia de processamento RAG: {e}")
        raise


def main(provider: LLMProvider = "openai", model="gpt-4o-mini", **llm_kwargs):
    logger.info("Iniciando sistema de perguntas e respostas")

    llm = initialize_llm(provider, model, **llm_kwargs)
    qa_chain = create_qa_chain(llm, retriever)

    while True:
        question = input("\nDigite sua pergunta (ou 'sair'): ")
        if question.lower() == "sair":
            logger.info("Encerrando o sistema")
            break

        query = f"query: {question}"
        logger.info(f"Processando pergunta: {question}")

        logger.debug("Recuperando documentos relevantes")

        retrieved_docs = retriever.get_relevant_documents(query)
        for i, doc in enumerate(retrieved_docs):
            logger.debug(f"--- Documento {i+1} ---")
            logger.debug(f"Fonte: {doc.metadata.get('source_doc', 'Desconhecido')}")
            logger.debug(f"Conteúdo: {doc.page_content}")

        try:
            logger.info("Gerando resposta...")

            result = qa_chain.invoke({"input": query})

            print("\n🧠 Resposta:")
            # The new chain returns the answer in the "answer" key
            print(result["answer"])
            logger.info("Resposta gerada com sucesso")

            print("\n📚 Fontes:")
            # The new chain returns source documents in the "context" key
            for doc in result["context"]:
                source = doc.metadata.get("source_doc", "Desconhecido")
                print(f"- {source}")
                logger.debug(f"Utilizando fonte: {source}")
        except Exception as e:
            logger.error(f"Erro ao processar pergunta: {e}")
            print(f"\n⚠️ Erro ao processar a pergunta: {e}")


if __name__ == "__main__":
    main()
