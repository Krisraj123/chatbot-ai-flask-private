
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

ASTRA_DB_APPLICATION_TOKEN = config['astra_db']['application_token']
ASTRA_DB_ID = config['astra_db']['db_id']
OPENAI_API_KEY = config['openai']['api_key']
GITHUB_TOKEN = config['github']['token']
AZURE_OPENAI_ENDPOINT = config['azure']['openai_endpoint']
AZURE_OPENAI_MODELNAME = config['azure']['model_name']
AZURE_OPENAI_EMBEDMODELNAME = config['azure']['embed_model_name']

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

    global chat_history
    global conversation_retrieval_chain

    output = conversation_retrieval_chain({"question":prompt, "chat_history":chat_history})
    answer = output["result"]

    chat_history.append((prompt,answer))
    return answer

init_llm()
path="Nithin_S_Resume1111.pdf"
process_document(path)

