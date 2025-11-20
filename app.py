import os
import pypdf
import chromadb
from chromadb.config import Settings
import ollama

model_name = "inclusion-bot"
MAX_HISTORY_TOKENS = 1000

# commented out all the debug prints
# for testing model connection
try:
    #print(f"Testing connection to model: {model_name}")
    stream_test = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": "hi there!"}],
        stream=True,
    )
    for chunk in stream_test:
        print(chunk["message"]["content"], end="", flush=True)
    # print("\nModel available.\n")

except ollama.ResponseError as e:
    if e.status_code == 404:
        print(f"Model '{model_name}' not found. Creating from Modelfile...")
        with open("Modelfile", "r", encoding="utf-8") as f:
            modelfile_text = f.read()
        ollama.create(model=model_name, modelfile=modelfile_text)
        print(f"Model '{model_name}' created. Rerun the script.")
        exit(0)
    else:
        print(f"Ollama error: {e}")
        exit(1)


# chromadb init
client = chromadb.PersistentClient(path="./chroma")


collection = client.get_or_create_collection(
    name="pdf_knowledge", metadata={"hnsw:space": "cosine"}
)


# pdf loader
def pdf_to_text(path):
    reader = pypdf.PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            # normalize spacing
            extracted = extracted.replace("\r", "")
            text += extracted.strip() + "\n"
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

        # better chunk splitting
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

        for i, chunk in enumerate(raw_chunks):
            try:
                emb = ollama.embed(model="nomic-embed-text", input=chunk)["embeddings"][
                    0
                ]
            except Exception as e:
                print(f"Embedding failed on chunk {i} of {file}: {e}")
                continue

            doc_id = f"{file}_chunk_{i}"

            collection.add(documents=[chunk], ids=[doc_id], embeddings=[emb])

    # print("PDF ingestion complete.\n")


# load pdfs on startup
load_pdfs()


# chat memory
conversation_history = []


# main RAG + chat function
def ask_with_context(question):
    # embed the question
    q_emb = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]

    # retrieve chunks
    result = collection.query(query_embeddings=[q_emb], n_results=3)

    retrieved_docs = result["documents"][0]
    context = "\n\n".join(retrieved_docs)

    # memory logic
    messages = []
    total_tokens = 0

    # reverse iteration
    for turn in reversed(conversation_history):
        turn_tokens = len(turn["user"].split()) + len(turn["bot"].split())

        if total_tokens + turn_tokens > MAX_HISTORY_TOKENS:
            break

        # prepend at once
        messages.insert(0, {"role": "assistant", "content": turn["bot"]})
        messages.insert(0, {"role": "user", "content": turn["user"]})

        total_tokens += turn_tokens

    # add current question with context
    prompt = f"""
Use this learning material to help the student:

{context}

Student question:
{question}
"""
    messages.append({"role": "user", "content": prompt})

    # ollama chat
    print("Bot:", end=" ", flush=True)
    full_response = ""

    try:
        stream = ollama.chat(model=model_name, messages=messages, stream=True)

        for chunk in stream:
            token = chunk["message"]["content"]
            full_response += token
            print(token, end="", flush=True)

    except Exception as e:
        print(f"\n\n!!! ERROR DURING STREAMING !!!\n{e}\n")
        return "Error: Could not generate response."

    print("\n")

    # save to memory
    conversation_history.append({"user": question, "bot": full_response})

    return full_response


# main loop
if __name__ == "__main__":
    # print("Chroma DB loaded successfully.")
    print("Chat with the bot. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye. Take care.")
            break

        ask_with_context(user_input)