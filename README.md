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
4. Set Ollama Host:
```powershell
$env:OLLAMA_HOST='http://localhost:11434' 
```
5. Create Python virtual environment:
```powershell
python -m venv 
```
6. Activate Python virtual environment:
```powershell
.venv\Scripts\Activate.ps1
```
7. Install dependencies:
```powershell
pip install -r requirements.txt   
```
8. Run python app:
```powershell
python app.py   
```
9. Go to:
```
http://localhost:5000
```
10. To stop:
```poweshell
ctrl + c
```


## DONE
- Custom Modelfile created (base: gemma3:4b)
- Frontend scaffolded
- PyPDF for PDF loading and chunking (/pdf)
- ChromaDB as a local vector database (/chroma)
- Added chat loop and memory context functions

## TODO
- Add PDF loader and integrate LangChain
- Test multi-lingual capabilities of gemma3:4b
- Add visual learning aid or image generation
- Text-to-speech integration


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