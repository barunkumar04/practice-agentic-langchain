from langchain_core.tools import Tool

# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    return now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format


def validate(to_validate: str) -> bool:
    # Define the user's question
    query = "valid?"

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.1},
    )
    relevant_docs = retriever.invoke(query)

    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        if to_validate in doc.page_content:
            print(f"Source: {doc.metadata['source']}\n")
            return True
    return False

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

    # for row in rows:
    #     print(row)

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
            name="Time",  # Name of the tool
            func=get_current_time,  # Function that the tool will execute
            # Description of the tool
            description="Useful for when you need to know the current time",
        ),
        Tool(
            name="Validate",  # Name of the tool
            func=validate,  # Function that the tool will execute
            # Description of the tool
            description="Useful for when you need to know valid country or currency",
        ),
        Tool(
            name="KafkaDeadletter",  # Name of the tool
            func=kafka_deadletter_check,  # Function that the tool will execute
            # Description of the tool
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