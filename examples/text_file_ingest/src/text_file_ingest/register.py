# flake8: noqa

# Import any tools which need to be automatically registered here
# register.py
import os
from text_file_ingest import text_file_ingest_function
from nat.core.builder import Builder
from nat.core.function_base import FunctionBaseConfig, register_function, FunctionInfo
from nat.data_models.embedder import EmbedderRef
from langchain_core.embeddings import Embeddings
from nat.data_models.llm import LLMFrameworkEnum

class TextFileIngestToolConfig(FunctionBaseConfig, name="text_file_ingest"):
    ingest_glob: str
    description: str
    chunk_size: int = 1024
    embedder_name: EmbedderRef = "nvidia/nv-embedqa-e5-v5"

@register_function(config_type=TextFileIngestToolConfig)
async def text_file_ingest_tool(config: TextFileIngestToolConfig, builder: Builder):
    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings: Embeddings = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    logger.info("Ingesting documents matching for the webpage: %s", config.ingest_glob)
    (ingest_dir, ingest_glob) = os.path.split(config.ingest_glob)
    loader = DirectoryLoader(ingest_dir, glob=ingest_glob, loader_cls=TextLoader)

    docs = [document async for document in loader.alazy_load()]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size)
    documents = text_splitter.split_documents(docs)
    vector = await FAISS.afrom_documents(documents, embeddings)

    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "text_file_ingest",
        config.description,
    )

    async def _inner(query: str) -> str:
        return await retriever_tool.arun(query)

    yield FunctionInfo.from_fn(_inner, description=config.description)