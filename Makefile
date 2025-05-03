# Variables
POETRY=poetry
PYTHON=$(POETRY) run python
STREAMLIT=$(POETRY) run streamlit

# Commands
.PHONY: install venv run test req clean

install:
	$(POETRY) install --with dev --no-root

venv:
	git submodule update --init --recursive
	$(POETRY) config virtualenvs.in-project true
	make install

run:
	$(STREAMLIT) run mentalhealth/app.py

test:
	$(POETRY) run pytest tests/

req:
	$(POETRY) export --without-hashes --with dev -f requirements.txt > requirements.txt

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +
	rm -rf .pytest_cache .mypy_cache .venv
