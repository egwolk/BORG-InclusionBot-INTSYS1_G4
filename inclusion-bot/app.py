import streamlit as st
import ollama
import tempfile
import pathlib
import pyttsx3

st.title("Borg: Inclusion Bot")

if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "inclusion-bot"
if "generating" not in st.session_state:
    st.session_state["generating"] = False
if "pending_user_prompt" not in st.session_state:
    st.session_state["pending_user_prompt"] = None

# Sidebar controls
st.sidebar.header("Settings")
enable_tts = st.sidebar.checkbox("Enable Text-to-Speech", value=True)
voice_rate = st.sidebar.slider("Voice rate", min_value=120, max_value=220, value=170)
voice_volume = st.sidebar.slider("Volume", min_value=0.0, max_value=1.0, value=1.0)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Re-render any audio previously generated for this assistant message
        if message.get("audio_bytes"):
            st.audio(message["audio_bytes"], format='audio/wav')



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

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Show an immediate thinking indicator before any tokens arrive
        
        with st.spinner("Borg is thinking…"):
            try:
                for chunk in ollama.chat(
                model=st.session_state["ollama_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                ],
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
        st.session_state.messages.append(msg_payload)

    # Clear pending and unlock input, then rerun to re-enable chat box
    st.session_state["pending_user_prompt"] = None
    st.session_state["generating"] = False
    st.rerun()