import pandas as pd
import gradio as gr
import logging
import sys
import torch
import chromadb

# LlamaIndex components for building the RAG pipeline
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.embeddings import resolve_embed_model
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging for better visibility, directing output to the console
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# --- Model Configuration ---
# This block sets up the embedding model and the LLM.
# It is designed to use free, open-source models that can run on a laptop with 12GB RAM.

# Define the embedding model. This converts text into numerical vectors.
# 'all-MiniLM-L6-v2' is a small, efficient model suitable for local use.
Settings.embed_model = resolve_embed_model("local:sentence-transformers/all-MiniLM-L6-v2")

# Define the LLM. This model generates the final answer.
# Phi-3-mini-4k-instruct is a small model designed to run on consumer hardware.
model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load the tokenizer and model.
# The tokenizer converts text to tokens, and the model generates the text.
# trust_remote_code=True is necessary for some custom Hugging Face model architectures.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Wrap the Hugging Face model in the LlamaIndex LLM class.
llm = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer,
    query_wrapper_prompt="<|user|>{query_str}</s><|assistant|>",
    generate_kwargs={"max_new_tokens": 128},
    device_map="auto"
)
Settings.llm = llm

# --- Data Loading and Indexing ---
def load_and_index_data(file_path):
    """
    Loads the CSV data from the specified path, converts each row into a LlamaIndex Document,
    and creates a vector store index for efficient querying.
    """
    print("Loading and indexing data...")
    # Read the CSV file into a pandas DataFrame.
    # low_memory=False is used to handle mixed data types in the large CSV file and prevent a DtypeWarning.
    df = pd.read_csv(file_path, low_memory=False)

    # Convert each car's data row into a single, comprehensive text string.
    # This prepares the data for the LLM to understand.
    documents = []
    for _, row in df.iterrows():
        # Correctly access the 'Make' and 'Modle' columns from your dataset.
        car_text = f"Car: {row['Make']} {row['Modle']}. \n"
        for col in df.columns:
            if pd.notna(row[col]) and col != 'Make' and col != 'Modle':
                car_text += f"{col.replace('_', ' ').title()}: {row[col]}. "
        documents.append(Document(text=car_text))
    
    # Set up a ChromaDB client to create a persistent vector store.
    # This stores the document embeddings on disk, so the expensive embedding step
    # doesn't need to be run every time the script starts.
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("car_specs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create the VectorStoreIndex.
    # This is the core RAG component that embeds the documents and stores them in ChromaDB.
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("Data loading and indexing complete.")
    return index

# --- Querying Logic ---
def query_data(index, query_text):
    """
    This function takes a user's query and uses the RAG query engine to get an answer.
    """
    # Create a query engine from the index.
    # This engine handles the retrieval of relevant car documents and passes them to the LLM
    # for response synthesis.
    query_engine = index.as_query_engine()

    # Execute the query.
    response = query_engine.query(query_text)
    return str(response)

# --- Gradio UI Setup ---
def run_app():
    """
    Main function to load data and launch the Gradio interface.
    """
    print("Setting up the RAG pipeline...")
    try:
        # The path to your CSV file is specified here.
        index = load_and_index_data("data/car_specifications.csv")
    except FileNotFoundError:
        print("Error: 'data/car_specifications.csv' not found.")
        return

    print("Launching Gradio interface...")
    # Create the Gradio interface.
    iface = gr.Interface(
        fn=lambda query: query_data(index, query),
        inputs=gr.Textbox(lines=2, placeholder="e.g., Which hypercar has a better 0-60mph time, the Bugatti Veyron or the Lamborghini Aventador?"),
        outputs="text",
        title="Exotic Car Comparator 🏎️",
        description="Ask me anything about the car specifications from the dataset!"
    )

    # Launch the web app.
    # share=False keeps the app local.
    # server_port is set to avoid conflicts.
    iface.launch(share=False, server_port=7860)
    print("Gradio app launched. Access at http://127.0.0.1:7860")

if __name__ == "__main__":
    run_app()
