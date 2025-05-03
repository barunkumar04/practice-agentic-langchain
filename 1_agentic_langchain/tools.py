import os
import sqlite3

from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# Load environment variables from .env file - Open API key
load_dotenv()

# Load the existing Chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "0_vector_store")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Check if the Chroma vector store already exists
if os.path.exists(persistent_directory):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=None)
else:
    raise FileNotFoundError(
        f"The directory {persistent_directory} does not exist. Please check the path."
    )

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}, #Top 3 hits
)

# Create a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)

def validate_iban(iban):
    """
    Validates an IBAN using the ISO 7064 MOD 97-10 algorithm.
    Example - Let's say we have the IBAN: DE41500105170123456789 
        1. Rearrange: The string becomes: 500105170123456789DE41.
        2. Convert letters: 5001051701234567891314.
        3. Integer Conversion: 5001051701234567891314.
        4. Modulo 97: The remainder of this integer divided by 97 is 1.
        5. Validation: Since the remainder is 1, the check digit test is passed.

    Args:
        iban: The IBAN string to validate.

    Returns:
        True if the IBAN is valid, False otherwise.
    """
    iban = iban.replace(" ", "").upper()
    if not iban.isalnum():
        return False
    if len(iban) < 15 or len(iban) > 34:
        return False

    moved_iban = iban[4:] + iban[:4]
    numeric_iban = "".join(str(ord(char) - 55) if char.isalpha() else char for char in moved_iban)

    return int(numeric_iban) % 97 == 1


def troubleshoot(troubleshoot: str) -> bool:
    
    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.1},
    )
    relevant_docs = retriever.invoke(troubleshoot)

    return relevant_docs

def kafka_deadletter_check(*args, **kwargs):
    # Connect to an in-memory database
    conn = sqlite3.connect(':memory:')

    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()

    # Create a table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kafka_deadletter (
            id INTEGER PRIMARY KEY,
            event TEXT NOT NULL
        )
    ''')
    #delete existing data
    cursor.execute("DELETE FROM kafka_deadletter")

    # Insert data into the table
    cursor.execute("INSERT INTO kafka_deadletter (id, event) VALUES (?, ?)", (101, 'CreateEvent'))
    cursor.execute("INSERT INTO kafka_deadletter (id, event) VALUES (?, ?)", (102, 'UpdateEvent'))

    # Commit the changes
    conn.commit()

    # Retrieve data from the table
    cursor.execute("SELECT * FROM kafka_deadletter")
    rows = cursor.fetchall()

    # Close the connection
    conn.close()
    return rows

def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        # Limit to two sentences for brevity
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."
        
def get_tools():
    tools = [
        Tool(
            name="Validate IBAN",  # Name of the tool
            func=validate_iban,  # Function that the tool will execute
            description="Useful for when you need to know the current time", # Description of the tool
        ),
        Tool(
            name="Troubleshoot", 
            func=troubleshoot, 
            description="Find troubleshooting step for given error",
        ),
        Tool(
            name="KafkaDeadletter",  
            func=kafka_deadletter_check,
            description="Useful for when you need to check kafka dead letter",
        ),
        Tool(
            name="Wikipedia",
            func=search_wikipedia,
            description="Useful for when you need to know information about a topic.",
        ),
        Tool(
            name="Answer Question",
            func=lambda input, **kwargs: rag_chain.invoke(
                {"input": input, "chat_history": kwargs.get("chat_history", [])}
            ),
            description="useful for when you need to answer questions about the context",
        ),
    ]

    return tools

def get_llm():
    return llm