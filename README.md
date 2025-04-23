# Agentic LangChain

Welcome to the demo of Agentic Langchain. This has workedout example to work with diffrent tools likes -
   - Code logic: Get current time
   - Validator: Use RAG vector store to validate country and currency
   - In memory table
   - Wikipedia
   - Chat GPT model


## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Poetry (Follow this [Poetry installation tutorial](https://python-poetry.org/docs/#installation) to install Poetry on your system)

### Installation

1. Clone the repository:

   ```bash
   <!-- TODO: UPDATE TO MY  -->
   git clone https://github.com/barunkumar04/practice-agentic-langchain
   cd practice-agentic-langchain
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install --no-root
   ```

3. Set up your environment variables:

   - Rename the `.env.example` file to `.env` and update the variables inside with your own values. Example:

   ```bash
   mv .env.example .env
   ```

4. Activate the Poetry shell to run the examples:

   ```bash
   poetry shell
   ```

5. Run to create Vector DB:

   ```bash
    python 0_helper_agent/0_vectorization_with_metadata.py
   ```

6. Run to experience Agentic Langchain:

   ```bash
    python 0_helper_agent/1_agent001.py
   ```

## 

## Credit
- https://brandonhancock.io/langchain-master-class
