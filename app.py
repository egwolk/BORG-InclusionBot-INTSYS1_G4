import streamlit as st
import ollama
import tempfile
import pathlib
import pyttsx3

st.title("Borg: Inclusion Bot")

if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "inclusion-bot"

# Sidebar controls
st.sidebar.header("Settings")
enable_tts = st.sidebar.checkbox("Enable Text-to-Speech", value=False)
voice_rate = st.sidebar.slider("Voice rate", min_value=120, max_value=220, value=170)
voice_volume = st.sidebar.slider("Volume", min_value=0.0, max_value=1.0, value=1.0)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
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
        message_placeholder.markdown(full_response)
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