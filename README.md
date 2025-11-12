# InclusionBot — Special Education Learning Assistant

Group 4 — Intelligent Systems (INTSYS) Final Project

A chatbot that gives simple explanations, TTS-friendly outputs, and visual aids for learners with special needs.

## SDG Coverage
- SDG 4 — Quality Education  
- SDG 10 — Reduced Inequalities

## Quick Start (Windows / PowerShell)
1. Build and start services:
```powershell
docker-compose up -d
``` 
2. Create the Ollama model from the Modelfile:
```powershell
docker exec -it ollama ollama create inclusion-bot -f /app/Modelfile
```
3. Open the frontend:
   - Go to http://localhost:3000
4. Make an account and start chatting.
5. To stop:
```powershell
docker compose down
```

Note: After making changes to the Modelfile or code, rebuild:
```powershell
docker-compose up --build -d
```

## DONE
- Custom Modelfile created (base: gemma3:4b)
- Frontend scaffolded
- Docker build and compose files configured

## TODO
- Add PDF loader and integrate LangChain
- Test multi-lingual capabilities of gemma3:4b
- Add visual learning aid or image generation
- Add automated tests and CI

## Resources
- [Ollama lab module 1.pdf](Resources/ollama%20lab%20module%201.pdf)

## Technologies used
- Python 3.11 (base image)
- ollama (Python client + Ollama engine)
- Docker
- Docker Compose
- Open Web UI (frontend)
- gemma3:4b (model base)
- pip (dependency management)

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