import streamlit as st
import ollama
import tempfile
import pathlib
import pyttsx3
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.errors import NotFoundError
# for console logging:
from pprint import pprint
import time 
import re

st.title("Borg: Inclusion Bot")

# flags for console logs
log_query_embedding = True  # query embedding logs
log_chromadb_results = True  # query results logs
log_context_retrieval = True  # context retrieval logs

if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "inclusion-bot"

# init chromadb client
if "chroma_client" not in st.session_state:
    st.session_state["chroma_client"] = chromadb.PersistentClient(path="./chroma_db")  # saves to ./chroma_db
    try:
        st.session_state["collection"] = st.session_state["chroma_client"].get_collection(name="pdf_chunks")
    except NotFoundError:  # collection doesn't exist yet
        st.session_state["collection"] = st.session_state["chroma_client"].create_collection(name="pdf_chunks")

# sidebar
st.sidebar.header("Settings")
enable_tts = st.sidebar.checkbox("Enable Text-to-Speech", value=False)
voice_rate = st.sidebar.slider("Voice rate", min_value=120, max_value=220, value=170)
voice_volume = st.sidebar.slider("Volume", min_value=0.0, max_value=1.0, value=1.0)

# pdf loader setting (advanced settings)
with st.sidebar.expander("Advanced Settings"):
    st.header("PDF Loading")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs"):
        if uploaded_files:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            progress_bar = st.progress(0)  # progress bar
            status_text = st.empty()  # status update
            total_files = len(uploaded_files)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}... (File {idx+1}/{total_files})")
                    
                    # extract text from pdf
                    pdf_reader = PdfReader(uploaded_file)
                    full_text = ""
                    for page in pdf_reader.pages:
                        full_text += page.extract_text()
                    
                    # chunker
                    chunks = text_splitter.split_text(full_text)
                    
                    # generate embeddings and store in chromadb
                    for i, chunk in enumerate(chunks):
                        embedding_response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
                        embedding = embedding_response.get("embedding", [])
                        if embedding:
                            st.session_state["collection"].add(
                                ids=[f"{uploaded_file.name}_chunk_{i}"],
                                embeddings=[embedding],
                                metadatas=[{"source": uploaded_file.name, "chunk_id": i}],
                                documents=[chunk]
                            )
                    
                    progress_bar.progress((idx + 1) / total_files)  # update progress
                    st.success(f"Processed {uploaded_file.name} successfully!")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
            
            status_text.text("All PDFs processed!")  # final status
            progress_bar.empty()  # hide progress bar after completion
        else:
            st.warning("No PDFs uploaded.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # retrieve chunks from chromadb
    context = ""
    try:
        if st.session_state["collection"].count() > 0:
            # embed the user's prompt
            query_embedding_response = ollama.embeddings(model="nomic-embed-text", prompt=prompt)
            query_embedding = query_embedding_response.get("embedding", [])

            # log the embedding
            if log_query_embedding:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Query Embedding:")
                print(" ".join(map(str, query_embedding))) 

            if query_embedding:
                # query the chromadb collection for relevant chunks
                results = st.session_state["collection"].query(
                    query_embeddings=[query_embedding],
                    n_results=3 # top 3 relevant chunks
                )
                
                # log chromadb results
                if log_chromadb_results:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ChromaDB Query Results:")
                    
                    # code for console output cleaning
                    cleaned_results = re.sub(r'(\n\s*)+', '\n', str(results))
                    pprint(cleaned_results)
                    
                    # extract documents from results
                    retrieved_chunks = results.get("documents", [[]])[0]
                    context = "\n\n".join(retrieved_chunks)

                # log the context + console cleanup
                if log_context_retrieval:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Context Retrieved:")
                    cleaned_context = re.sub(r'(\n\s*)+', '\n', context)
                    pprint(cleaned_context)
    except Exception as e:
        st.warning(f"Retrieval error: {e}")

    # add context to messages if available
    messages_for_ollama = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]
    if context:
        messages_for_ollama.insert(0, {"role": "system", "content": f"Relevant context from documents:\n{context}"})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for chunk in ollama.chat(
                model=st.session_state["ollama_model"],
                messages=messages_for_ollama,
                stream=True,
            ):
                content_piece = chunk.get("message", {}).get("content", "")
                if content_piece:
                    full_response += content_piece
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response) # remove symbol after completion

        except Exception as e:
            st.error(f"Ollama chat error: {e}")
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Optional TTS playback
        if enable_tts and full_response.strip():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', voice_rate)
                engine.setProperty('volume', voice_volume)

                # Use default voice; users can configure OS voices
                with tempfile.TemporaryDirectory() as tmpdir:
                    wav_path = pathlib.Path(tmpdir) / "response.wav"
                    engine.save_to_file(full_response, str(wav_path))
                    engine.runAndWait()
                    # Read and play audio
                    audio_bytes = wav_path.read_bytes()
                    st.audio(audio_bytes, format='audio/wav')
            except Exception as e:
                st.warning(f"TTS error: {e}")
