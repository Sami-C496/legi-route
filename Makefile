.PHONY: install test run cli download index eval lint

install:
	poetry install

test:
	poetry run pytest tests/ -v

run:
	poetry run streamlit run src/app.py

cli:
	poetry run python main.py

download:
	poetry run python src/ingestion/download.py

index:
	poetry run python src/ingestion/indexing.py

eval:
	poetry run python eval/eval_ragas.py

lint:
	poetry run black src/ tests/
