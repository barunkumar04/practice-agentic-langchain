import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader

#loading from .env variables. Eg - API key
load_dotenv()


# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
knowledge_base = os.path.join(current_dir, "..","0_knowledge_base")
db_dir = os.path.join(current_dir, "..", "0_vector_store")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print("\n")
print(f"Knowledge base: {knowledge_base}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("\nPersistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(knowledge_base):
        raise FileNotFoundError(
            f"The directory {knowledge_base} does not exist. Please check the path."
        )

    # List all text files in the directory
    validator_files = [f for f in os.listdir(knowledge_base) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    
    # Loading from files 
    for validator_file in validator_files:
        file_path = os.path.join(knowledge_base, validator_file)
        loader = TextLoader(file_path)
        files = loader.load()
        for file in files:
            # Add metadata to each document indicating its source
            file.metadata = {"source": validator_file}
            documents.append(file)

    # Loading from website
    url = "https://www.globalshares.com"

    # Create a loader for web content
    web_loader = WebBaseLoader(url)
    web_docs = web_loader.load()
    for web_doc in web_docs:
            # Add metadata to each document indicating its source
            web_doc.metadata = {"source": url}
            documents.append(web_doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n")
    print(f"Number of document chunks: {len(docs)}.")

    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Update to a valid embedding model if needed
    print("Finished creating embeddings.")

    # Create the vector store and persist it
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("Finished creating and persisting vector store.")

else:
    print("\nVector store already exists. No need to initialize.")
