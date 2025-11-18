import os
import pypdf
import chromadb
from chromadb.config import Settings
import ollama

model_name = 'inclusion-bot'
MAX_HISTORY_TOKENS = 1000  # estimated memory limit for history

# connect to ollama and create model file if missing
try:
    stream_test = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': 'hi there!'}],
        stream=True
    )
    print(f"Successfully connected to {model_name}\n")
    for chunk in stream_test:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")
except ollama.ResponseError as e:
    if e.status_code == 404:
        print(f"Model '{model_name}' not found. Creating from Modelfile...")
        with open('Modelfile', 'r', encoding='utf-8') as f:
            modelfile_text = f.read()
        ollama.create(model=model_name, modelfile=modelfile_text)
        print(f"Model '{model_name}' created. Rerun the script.")
        exit(0)
        
    else:
        print(f"An error occurred: {e}")
        exit(1)

# chromadb init
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma"
))

collection = client.get_or_create_collection(
    name="pdf_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# pdf handler and chunking
def pdf_to_text(path):
    reader = pypdf.PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def load_pdfs():
    pdf_dir = "./pdfs"
    if not os.path.exists(pdf_dir):
        print("No PDFs found. Create a folder named 'pdfs' and add files there.")
        return

    for file in os.listdir(pdf_dir):
        if not file.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(pdf_dir, file)
        print(f"Ingesting {file} ...")

        text = pdf_to_text(full_path)
        chunks = text.split("\n\n")

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                continue

            emb = ollama.embed(
                model="nomic-embed-text",
                input=chunk
            )["embeddings"][0]

            doc_id = f"{file}_chunk_{i}"

            collection.add(
                documents=[chunk],
                ids=[doc_id],
                embeddings=[emb]
            )

    client.persist()

# load pdfs at startup
load_pdfs()

# load memory
conversation_history = []

def ask_with_context(question):
    # embed the question
    q_emb = ollama.embed(
        model="nomic-embed-text",
        input=question
    )["embeddings"][0]

    # retrieve relevant pdf chunks
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=3
    )
    context = "\n\n".join(result["documents"][0])

    # build messages for ollama with memory
    messages = []
    total_tokens = 0
    for turn in reversed(conversation_history):
        # approx token count: 1 word / token
        turn_tokens = len(turn['user'].split()) + len(turn['bot'].split())
        if total_tokens + turn_tokens > MAX_HISTORY_TOKENS:
            break  # stop adding older turns
        messages.insert(0, {"role": "user", "content": turn['user']})
        messages.insert(1, {"role": "assistant", "content": turn['bot']})
        total_tokens += turn_tokens

    # add the current question with context
    prompt = f"""
Use this learning material to help the student:

{context}

Student question:
{question}
"""
    messages.append({"role": "user", "content": prompt})

    stream = ollama.chat(
        model=model_name,
        messages=messages,
        stream=True
    )

    print("Bot:", end=" ", flush=True)
    full_response = ""
    for chunk in stream:
        token = chunk['message']['content']
        full_response += token
        print(token, end='', flush=True)
    print("\n")

    # save this turn in memory
    conversation_history.append({"user": question, "bot": full_response})
    return full_response

# multi turn chat loop
if __name__ == "__main__":
    print("Chroma vector DB loaded successfully.")
    print("Chat with the bot. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye. Take care.")
            break

        ask_with_context(user_input)