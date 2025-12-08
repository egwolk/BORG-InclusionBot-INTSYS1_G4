# InclusionBot — Special Education Learning Assistant

Group 4 — Intelligent Systems (INTSYS) Final Project

A chatbot that gives simple explanations, TTS-friendly outputs, and visual aids for learners with special needs.

## SDG Coverage
- SDG 4 — Quality Education  
- SDG 10 — Reduced Inequalities

## Quick Start (Windows / PowerShell)
1. Install Ollama and python.

2. Pull the base model and embedding model:
```powershell
 ollama pull gemma3:4b 
 ollama pull nomic-embed-text
```
3. Build the Modelfile:
```powershell
ollama create inclusion-bot -f Modelfile 
```
4. Create Python virtual environment:
```powershell
python -m venv venv
```
5. Activate Python virtual environment:
```powershell
venv\Scripts\Activate
```
6. Install dependencies:
```powershell
pip install -r requirements.txt   
```
7. Set YouTube API Key:
```powershell
$env:YOUTUBE_API_KEY = "CHECK_TEXT.txt_AND_PASTE_HERE" 
```
If this doesn't work, create a new API key with this [tutorial](https://www.youtube.com/watch?v=TaHFJ03Vox0)

8. Run python app:
```powershell
streamlit run app.py   
```
9. Go to:
```
http://localhost:8501
```
Note: If the embed gives an `error:153`, disable UBlockOrigin or try a different browser.
10. To stop:
```poweshell
ctrl + c
```


## DONE
- Custom Modelfile created (base: gemma3:4b)
- Frontend scaffolded
- Text-to-speech integration
- Add visual learning aid or image generation
- Added chat loop and memory context functions
- PyPDF for PDF loading and chunking (/pdf)
- ChromaDB as a local vector database (/chroma)

## TODO
- None



## Technologies used
- Python 3.11 (base image)
- ollama (Python client + Ollama engine)
- Flask (frontend)
- gemma3:4b (model base)
- chromadb
- pypdf
- nomic-embed-text (embedding model)

## Notes
- The Modelfile lives at `./Modelfile`.
- The app entrypoint is `app.py`.
- requirements.txt currently contains `ollama`. Add other deps as needed.

## Group Members (Add your link!):
- De Guzman, Jonah Andre P. [@jdgmn](https://github.com/jdgmn)
- Covacha, Erin Drew C. [@egwolk](https://github.com/egwolk)
- Brieta, Eleandro Frederick G.
- Garcia, Raphael J.
- Kanapi, Ceolo Diane A.
- Oppas, Roldan P.

---
## Made with 💗