import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    show_progress_bar=False, chunk_size=50, 
    retry_min_seconds=10
)

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

# Clear the index if it's already populated to avoid duplicate documents
doc_count = vectorstore.index.describe_index_stats()["total_vector_count"]

if doc_count > 0:
    print(f"Found {doc_count} documents")
    print("Clearing index")
    vectorstore.delete(delete_all=True)

vectorstore.add_documents(doc_splits)

retriever = vectorstore.as_retriever()