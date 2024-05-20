import logging
import sys

# Uncomment to see debug logs
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import phoenix as px
from llama_index.vector_stores.milvus import MilvusVectorStore
from IPython.display import Markdown, display
import textwrap
from llama_index.readers.file import CSVReader, PagedCSVReader
from pathlib import Path
import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import sys

sys.path.append("../")
from config import GOOGLE_API_KEY, PINECONE_API_KEY
import os

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
from llama_index.core import VectorStoreIndex, get_response_synthesizer, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)
from langchain.agents import tool
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.prompts import display_prompt_dict

from pinecone import Pinecone
from pinecone import ServerlessSpec

pinecoin = Pinecone(api_key=PINECONE_API_KEY)

pinecone_index = pinecoin.Index("shopping-assistant")
from llama_index.vector_stores.pinecone import PineconeVectorStore

from llama_index.core import StorageContext

px.launch_app()
import llama_index.core

llama_index.core.set_global_handler("arize_phoenix")

# node_parser = SentenceSplitter(chunk_size=4096)

# nodes = node_parser.get_nodes_from_documents(documents)
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo


vector_store_info = VectorStoreInfo(
    content_info="Danh sách sản phẩm của cửa hàng cellphones",
    metadata_info=[
        MetadataInfo(
            name="PRODUCT_NAME",
            type="string",
            description=(
                "Đây là tên của sản phẩm. Thường là một chuỗi văn bản mô tả sản phẩm một cách ngắn gọn và rõ ràng"
            ),
        ),
        MetadataInfo(
            name="PRICE_REMAINING",
            type="int",
            description=(
                "Đây là giá sản phẩm hiện tại sau khi đã giảm giá hoặc có thể là giá gốc nếu sản phẩm không được giảm giá. Thường là một số nguyên hoặc số thực biểu thị tiền tệ."
            ),
        ),
        # MetadataInfo(
        #     name="PRICE_PERCENTAGE_REDUCE",
        #     type="string",
        #     description=(
        #         "Đây là phần trăm giảm giá so với giá ban đầu của sản phẩm. Thường là một số thực biểu thị phần trăm giảm giá."
        #     ),
        # ),
        MetadataInfo(
            name="PRODUCT_IMAGE",
            type="string",
            description=(
                "Đây là liên kết đến hình ảnh của sản phẩm. Thường là một URL dẫn đến hình ảnh sản phẩm."
            ),
        ),
        MetadataInfo(
            name="PRODUCT_LINK",
            type="string",
            description=(
                "Đây là liên kết đến trang chi tiết sản phẩm. Thường là một URL dẫn đến trang sản phẩm trên trang web bán hàng."
            ),
        ),
        MetadataInfo(
            name="Review",
            type="string",
            description=(
                "Đây là một chuỗi văn bản mô tả đánh giá hoặc nhận xét từ người dùng."
            ),
        ),
    ],
)


class Retrieval:
    def __init__(self) -> None:
        """
        This class is used to retrieve the index and query engine for the shopping assistant application
        """
        
    def load_documents(self, path):
        # load documents
        df = pd.read_csv(path)
        df["PRICE_REMAINING"] = (
            df["PRICE_REMAINING"].str.replace(".", "").str.replace("₫", "").astype(int)
        )
        documents = []
        for index, value in df.iterrows():
            document = Document(
                text=value["PRODUCT_INFOMATION_DETAIL"],
                metadata={
                    "PRODUCT_NAME": value["PRODUCT_NAME"],
                    "PRICE_REMAINING": value["PRICE_REMAINING"],
                    # "PRICE_INITIAL": value["PRICE_INITIAL"],
                    # "PRICE_PERCENTAGE_REDUCE": value["PRICE_PERCENTAGE_REDUCE"],
                    "PRODUCT_IMAGE": value["PRODUCT_IMAGE"],
                    "PRODUCT_LINK": value["PRODUCT_LINK"],
                    "Review": value["Review"],
                },
            )
            documents.append(document)

    def create_index(self, documents, embed_model):
        from llama_index.core import StorageContext

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index, namespace="text"
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
            embed_model=embed_model,
        )
        return index

    def load_index(self, embed_model):
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index, namespace="text"
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, show_progress=True, embed_model=embed_model
        )
        return index

    def create_query_engine(self, index, llm):
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        # configure retriever
        retriever = VectorIndexAutoRetriever(
            index,
            vector_store_info=vector_store_info,
            llm=llm,
            verbose=True,
            callback_manager=callback_manager,
            similarity_top_k=10,
            empty_query_top_k=10,
            default_empty_query_vector=[0] * 1536,
        )

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=llm, response_mode="compact", callback_manager=callback_manager
        )

        # assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager,
            # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        )

        return query_engine
