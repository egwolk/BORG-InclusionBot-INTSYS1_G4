import ollama
model_name = 'inclusion-bot'

try:
    stream = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': 'hi there!'}],
        stream=True
    )
    print(f"successfully connected to {model_name}\n")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
except ollama.ResponseError as e:
    if e.status_code == 404:
        print(f"Model '{model_name}' not found. Creating from Modelfile...")
        with open('Modelfile', 'r', encoding='utf-8') as f:
            modelfile_text = f.read()
        ollama.create(model=model_name, modelfile=modelfile_text)
        print(f"Model '{model_name}' created. Rerun the script.")
    else:
        print(f"An error occurred: {e}")