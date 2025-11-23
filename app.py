from flask import Flask, render_template, request, jsonify
import ollama
import os
import pypdf
import chromadb


app = Flask(__name__)
MODEL = "inclusion-bot"

# --- ChromaDB setup and PDF ingestion ---------------------------------
# persistent client stores DB on disk under ./chroma
client = chromadb.PersistentClient(path="./chroma")

collection = client.get_or_create_collection(
    name="pdf_knowledge", metadata={"hnsw:space": "cosine"}
)


def pdf_to_text(path):
    reader = pypdf.PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            extracted = extracted.replace("\r", "")
            text += extracted.strip() + "\n"
    return text


def chunk_text(text, max_words=250):
    """Very simple chunker that splits on paragraphs then limits by word count."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        if not current:
            current = p
            continue

        # if adding paragraph would exceed size, push current
        if len((current + " " + p).split()) > max_words:
            chunks.append(current)
            current = p
        else:
            current = current + "\n\n" + p

    if current:
        chunks.append(current)

    return chunks


def load_pdfs(pdf_dir="./pdfs"):
    if not os.path.exists(pdf_dir):
        print("No PDFs found. Create a folder named 'pdfs' and add files there.")
        return

    for file in os.listdir(pdf_dir):
        if not file.lower().endswith(".pdf"):
            continue

        full_path = os.path.join(pdf_dir, file)
        print(f"Ingesting {file} ...")

        text = pdf_to_text(full_path)
        raw_chunks = chunk_text(text, max_words=300)

        for i, chunk in enumerate(raw_chunks):
            try:
                emb = ollama.embed(model="nomic-embed-text", input=chunk)["embeddings"][0]
            except Exception as e:
                print(f"Embedding failed on chunk {i} of {file}: {e}")
                continue

            doc_id = f"{file}_chunk_{i}"

            # avoid duplicate ids: drop if exists
            try:
                collection.add(documents=[chunk], ids=[doc_id], embeddings=[emb])
            except Exception:
                # fallback: try to upsert by removing first
                try:
                    collection.delete(ids=[doc_id])
                except Exception:
                    pass
                collection.add(documents=[chunk], ids=[doc_id], embeddings=[emb])


# load PDFs at startup (non-blocking simple approach)
try:
    load_pdfs()
except Exception as e:
    print(f"PDF ingestion at startup failed: {e}")

# --- Conversation memory ------------------------------------------------
conversation_history = []
MAX_HISTORY_TOKENS = 1000


def ask_with_context(question, top_k=3):
    # embed the question
    try:
        q_emb = ollama.embed(model="nomic-embed-text", input=question)["embeddings"][0]
    except Exception as e:
        return f"Error embedding query: {e}"

    # retrieve similar chunks
    try:
        result = collection.query(query_embeddings=[q_emb], n_results=top_k)
        retrieved_docs = result.get("documents", [[]])[0]
    except Exception:
        retrieved_docs = []

    context = "\n\n".join(retrieved_docs) if retrieved_docs else ""

    # compact conversation history (word-count based simple approximation)
    messages = []
    total_tokens = 0

    for turn in reversed(conversation_history):
        turn_tokens = len(turn["user"].split()) + len(turn["bot"].split())
        if total_tokens + turn_tokens > MAX_HISTORY_TOKENS:
            break
        messages.insert(0, {"role": "assistant", "content": turn["bot"]})
        messages.insert(0, {"role": "user", "content": turn["user"]})
        total_tokens += turn_tokens

    # Build user prompt that contains the retrieved context
    prompt = f"Use this learning material to help the student:\n\n{context}\n\nStudent question:\n{question}"
    messages.append({"role": "user", "content": prompt})

    try:
        resp = ollama.chat(model=MODEL, messages=messages, stream=False)
        content = resp.get("message", {}).get("content", "")
    except Exception as e:
        content = f"Error generating response: {e}"

    # save to memory
    conversation_history.append({"user": question, "bot": content})

    return content


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    msg = data.get("message", "")
    if not msg:
        return jsonify({"error": "no message"}), 400

    try:
        reply = ask_with_context(msg)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)