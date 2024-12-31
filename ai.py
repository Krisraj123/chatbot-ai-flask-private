
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from typing import List, Union
from datasets import load_dataset
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.base import VectorStoreRetriever
# Support for dataset retrieval with Hugging Face
from azure.core.credentials import AzureKeyCredential
from langchain.chains import RetrievalQA

from langchain.text_splitter import CharacterTextSplitter
# With CassIO, the engine powering the Astra DB integration in LangChain,
# you will also initialize the DB connection:
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
import json
import cassio

with open('config.json', 'r') as f:
    config = json.load(f)

ASTRA_DB_APPLICATION_TOKEN = "AstraCS:AcdzriRNttodLZfRbrcaUDlQ:d5d4203e68f970bab60ade7fc18a4cf76eaab9bf153d703d2eb1c12ab7306e15"
ASTRA_DB_ID = "d0bebb10-9f89-4482-966b-8fc24a086d70"
OPENAI_API_KEY = "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"
GITHUB_TOKEN = "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"
AZURE_OPENAI_ENDPOINT = "https://models.inference.ai.azure.com"
AZURE_OPENAI_MODELNAME = "gpt-4o"
AZURE_OPENAI_EMBEDMODELNAME = "text-embedding-3-large"

conversation_retrieval_chain = None
chat_history = []
llm = None
embedding = None
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

class AzureOpenAIEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client
        self.model_name = AZURE_OPENAI_EMBEDMODELNAME  # Store model name

    def embed_query(self, text: str):
        """Embed a query."""
        response = self.client.embed(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list):
        """Embed a list of documents."""
        response = self.client.embed(
            input=texts,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

def init_llm():
    global llm , embedding
    llm = OpenAI( base_url=AZURE_OPENAI_ENDPOINT, api_key=GITHUB_TOKEN, model=AZURE_OPENAI_MODELNAME)
    embedding = EmbeddingsClient(
        endpoint=AZURE_OPENAI_ENDPOINT,
        credential=AzureKeyCredential(GITHUB_TOKEN),
        model = AZURE_OPENAI_EMBEDMODELNAME
    )

def process_document(document_path):
    init_llm()
    global conversation_retrieval_chain
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap  = 200,
    )
    raw_text ="".join([doc.page_content for doc in documents])
    texts = text_splitter.split_text(raw_text)
    custom_embedding = AzureOpenAIEmbeddings(embedding)
    astra_vector_store = Cassandra(
      embedding=custom_embedding,
      table_name="qa_mini_demo",
      session=None,
      keyspace=None,
    )
    astra_vector_store.add_texts(texts[:500])
    retriever = VectorStoreRetriever(vectorstore = astra_vector_store, search_type="mmr", search_kwargs={'k': 1, 'lambda_mult': 0.25})
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents = False,
        input_key = "question"
    )

def process_prompt(prompt):
    init_llm()
    global chat_history
    global conversation_retrieval_chain

    output = conversation_retrieval_chain({"question":prompt, "chat_history":chat_history})
    answer = output["result"]

    chat_history.append((prompt,answer))
    return answer


