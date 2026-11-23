## indexer.py
uv run indexer.py

## rag.py
uv run rag.py --texte "une bouteille d'huile d'olive"

uv run rag.py --texte "a cat food pouch"

uv run rag.py --texte "des chaussettes"


uv run rag.py image.jpg
uv run rag.py image.jpg --gemini
uv run rag.py image.jpg "Est-ce que c'est en promotion ?"

## app.py
uv run streamlit run app.py
http://localhost:8501