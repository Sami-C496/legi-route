.PHONY: install test run cli download index eval lint release

VERSION := $(shell python -c "exec(open('src/version.py').read()); print(__version__)")

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

release:
	@echo "Tagging v$(VERSION)..."
	git tag -a v$(VERSION) -m "v$(VERSION)"
	git push origin v$(VERSION)
	gh release create v$(VERSION) --title "v$(VERSION)" --notes-file CHANGELOG.md
	@echo "Release v$(VERSION) published."
