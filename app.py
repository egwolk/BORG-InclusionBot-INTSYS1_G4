import streamlit as st
import ollama
import tempfile
import pathlib
import pyttsx3
import os
import requests
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.errors import NotFoundError
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
if "generating" not in st.session_state:
    st.session_state["generating"] = False
if "pending_user_prompt" not in st.session_state:
    st.session_state["pending_user_prompt"] = None

# init chromadb client
if "chroma_client" not in st.session_state:
    st.session_state["chroma_client"] = chromadb.PersistentClient(path="./chroma_db")  # saves to ./chroma_db
    try:
        st.session_state["collection"] = st.session_state["chroma_client"].get_collection(name="pdf_chunks")
    except NotFoundError:  # collection doesn't exist yet
        st.session_state["collection"] = st.session_state["chroma_client"].create_collection(name="pdf_chunks")

# Sidebar controls
st.sidebar.header("Settings")
enable_tts = st.sidebar.checkbox("Enable Text-to-Speech", value=True)
voice_rate = st.sidebar.slider("Voice rate", min_value=120, max_value=220, value=170)
voice_volume = st.sidebar.slider("Volume", min_value=0.0, max_value=1.0, value=1.0)

# YouTube suggestion controls
st.sidebar.subheader("YouTube suggestions")
enable_youtube = st.sidebar.checkbox("Suggest YouTube videos", value=True)
yt_api_key = os.getenv("YOUTUBE_API_KEY", "")

# pdf loader setting (advanced settings)
with st.sidebar.expander("Advanced Settings"):
    st.header("PDF Loading")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    chunk_size = st.slider("Chunk size", min_value=200, max_value=1500, value=500, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=50, step=50)
    
    if st.button("Process PDFs"):
        if uploaded_files:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"
                    
                    if not full_text.strip():
                        st.warning(f"{uploaded_file.name} appears to be empty or couldn't be read.")
                        continue
                    
                    # chunker
                    chunks = text_splitter.split_text(full_text)
                    total_chunks = len(chunks)
                    status_text.text(f"Processing {uploaded_file.name}... ({total_chunks} chunks, File {idx+1}/{total_files})")
                    
                    # generate embeddings and store in chromadb with rate limiting
                    successful_chunks = 0
                    for i, chunk in enumerate(chunks):
                        try:
                            # Skip empty chunks
                            if not chunk.strip():
                                continue
                            
                            # Add a small delay to avoid overwhelming Ollama
                            if i > 0 and i % 10 == 0:
                                time.sleep(0.5)  # brief pause every 10 chunks
                            
                            embedding_response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
                            embedding = embedding_response.get("embedding", [])
                            
                            if embedding:
                                # Use a unique ID that includes timestamp to avoid collisions
                                chunk_id = f"{uploaded_file.name}_{int(time.time())}_{i}"
                                st.session_state["collection"].add(
                                    ids=[chunk_id],
                                    embeddings=[embedding],
                                    metadatas=[{"source": uploaded_file.name, "chunk_id": i}],
                                    documents=[chunk]
                                )
                                successful_chunks += 1
                                
                                # Update progress for large files
                                if i % 5 == 0:
                                    status_text.text(f"Processing {uploaded_file.name}... ({successful_chunks}/{total_chunks} chunks done)")
                        
                        except Exception as chunk_error:
                            print(f"Error embedding chunk {i} from {uploaded_file.name}: {chunk_error}")
                            # Continue processing other chunks even if one fails
                            continue
                    
                    progress_bar.progress((idx + 1) / total_files)  # update progress
                    st.success(f"Processed {uploaded_file.name}: {successful_chunks}/{total_chunks} chunks successfully embedded!")
                
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    print(f"Full error for {uploaded_file.name}: {e}")
            
            status_text.text("All PDFs processed!")  # final status
            time.sleep(2)
            progress_bar.empty()  # hide progress bar after completion
            status_text.empty()
        else:
            st.warning("No PDFs uploaded.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_youtube_response" not in st.session_state:
    st.session_state["awaiting_youtube_response"] = False
if "pending_youtube_query" not in st.session_state:
    st.session_state["pending_youtube_query"] = None

@st.cache_data(show_spinner=False, ttl=600)
def search_youtube(query: str, api_key: str, max_results: int = 3):
    if not api_key or not query.strip():
        return []
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "safeSearch": "strict",
        "videoEmbeddable": "true",
        "relevanceLanguage": "en",
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    videos = []
    for item in data.get("items", []):
        vid = item.get("id", {}).get("videoId")
        sn = item.get("snippet", {})
        if not vid:
            continue
        videos.append({
            "title": sn.get("title", "Untitled"),
            "channel": sn.get("channelTitle", "Unknown channel"),
            "thumbnail": (sn.get("thumbnails", {}) or {}).get("medium", {}).get("url"),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return videos

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("audio_bytes"):
            st.audio(message["audio_bytes"], format='audio/wav')
        yt_videos_prev = message.get("yt_videos") or []
        if yt_videos_prev:
            st.markdown("Recommended videos:")
            for v in yt_videos_prev:
                st.video(v["url"])
                st.markdown(f"**{v['title']}**  \nChannel: {v['channel']}")



# Chat input — disabled when generating
prompt_input = st.chat_input(
    "What's up?",
    disabled=st.session_state.get("generating", False),
)

# On new input, store pending and force rerun to immediately disable input
if prompt_input and not st.session_state.get("generating", False):
    st.session_state["pending_user_prompt"] = prompt_input
    st.session_state["generating"] = True
    st.rerun()

# If there is a pending prompt, process it now
if st.session_state.get("pending_user_prompt"):
    user_text = st.session_state["pending_user_prompt"]

    with st.chat_message("user"):
        st.markdown(user_text)
    st.session_state.messages.append({"role": "user", "content": user_text})

    # Check if user is responding to YouTube video offer
    youtube_confirmation_keywords = ["yes", "yeah", "yep", "sure", "ok", "okay", "yes please", "show me"]
    youtube_decline_keywords = ["no", "nope", "nah", "no thanks", "not now", "skip"]
    is_youtube_response = st.session_state.get("awaiting_youtube_response", False)
    
    # retrieve chunks from chromadb
    context = ""
    
    # Skip RAG for identity/greeting questions to preserve model's personality
    identity_keywords = ["who are you", "what is your name", "introduce yourself", 
                         "what's your name", "who r u", "your name"]
    skip_rag = any(keyword in user_text.lower() for keyword in identity_keywords) or is_youtube_response
    
    if not skip_rag:
        try:
            if st.session_state["collection"].count() > 0:
                # embed the user's prompt
                query_embedding_response = ollama.embeddings(model="nomic-embed-text", prompt=user_text)
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
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Skipping RAG for identity question")

    # add context to messages if available
    messages_for_ollama = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]
    if context:
        messages_for_ollama.insert(0, {"role": "system", "content": f"Relevant context from documents:\n{context}"})

    # Handle YouTube confirmation/decline
    yt_videos = []
    if is_youtube_response:
        user_text_lower = user_text.lower().strip()
        
        if any(keyword in user_text_lower for keyword in youtube_confirmation_keywords):
            # User wants videos
            with st.chat_message("assistant"):
                if enable_youtube and yt_api_key and st.session_state.get("pending_youtube_query"):
                    try:
                        yt_videos = search_youtube(st.session_state["pending_youtube_query"], yt_api_key, 3)[:3]
                        if yt_videos:
                            st.markdown("Here are some videos that might help:")
                            for v in yt_videos[:3]:
                                st.video(v["url"])
                                st.markdown(f"**{v['title']}**  \nChannel: {v['channel']}")
                            full_response = "Here are some videos that might help:"
                        else:
                            st.markdown("I couldn't find any videos right now. Try asking me something else!")
                            full_response = "I couldn't find any videos right now. Try asking me something else!"
                    except Exception as e:
                        st.info(f"I had trouble finding videos: {e}")
                        full_response = "I had trouble finding videos right now."
                else:
                    st.markdown("Video suggestions are turned off right now.")
                    full_response = "Video suggestions are turned off right now."
            
            audio_bytes = None
            st.session_state["awaiting_youtube_response"] = False
            st.session_state["pending_youtube_query"] = None
            
        elif any(keyword in user_text_lower for keyword in youtube_decline_keywords):
            # User declined videos
            with st.chat_message("assistant"):
                st.markdown("Okay! Let me know if you need anything else.")
                full_response = "Okay! Let me know if you need anything else."
            
            audio_bytes = None
            st.session_state["awaiting_youtube_response"] = False
            st.session_state["pending_youtube_query"] = None
        else:
            # User said something else, treat as new question
            st.session_state["awaiting_youtube_response"] = False
            st.session_state["pending_youtube_query"] = None
            is_youtube_response = False
    
    # Normal conversation flow (not a YouTube response)
    if not is_youtube_response:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Show an immediate thinking indicator before any tokens arrive
            
            with st.spinner("Borg is thinking…"):
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
                except Exception as e:
                    st.error(f"Ollama chat error: {e}")
            # Replace the thinking indicator with the final response
            message_placeholder.markdown(full_response)
            
            # After explanation, offer YouTube videos if enabled
            if enable_youtube and yt_api_key and user_text.strip() and not any(keyword in user_text.lower() for keyword in identity_keywords):
                youtube_offer = "\n\nWould you like me to show you some videos about this topic?"
                message_placeholder.markdown(full_response + youtube_offer)
                full_response += youtube_offer
                
                # Set flags to await user's response
                st.session_state["awaiting_youtube_response"] = True
                st.session_state["pending_youtube_query"] = user_text
            
            audio_bytes = None
            # Optional TTS playback and persistence
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

        # Persist the assistant message (including audio, if any) so it survives reruns
        msg_payload = {"role": "assistant", "content": full_response}
        if audio_bytes:
            msg_payload["audio_bytes"] = audio_bytes
        if yt_videos:
            msg_payload["yt_videos"] = yt_videos
        st.session_state.messages.append(msg_payload)

    # Clear pending and unlock input, then rerun to re-enable chat box
    st.session_state["pending_user_prompt"] = None
    st.session_state["generating"] = False
    st.rerun()