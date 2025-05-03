# Agentic LangChain

Welcome to the demo of Agentic Langchain. This has workedout example to work with diffrent tools likes -
   - Code logic: Validate IBAN
   - RAG: Fine troubleshooting steps for error
   - In memory table: Find row info
   - Wikipedia
   - Chat GPT model: For any question answer


## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Poetry (Follow this [Poetry installation tutorial](https://python-poetry.org/docs/#installation) to install Poetry on your system)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/barunkumar04/practice-agentic-langchain
   cd practice-agentic-langchain
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry env activate
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
5. Create folder '0_vector_store', in paralled to '0_knowledge_base'
6. Run to create Vector DB:

   ```bash
    python 1_agentic_langchain/create_vector_store.py
   ```

7. Run to experience Agentic Langchain:

   ```bash
    python 1_agentic_langchain/agent001.py
   ```


## Credit
- https://brandonhancock.io/langchain-master-class
