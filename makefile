ASS1 := ass1
ASS2 := ass2

.PHONY: install
install:
	poetry install

.PHONY: update
update:
	poetry update

.PHONY: lint
lint:
	poetry run flake8


.PHONY: format
format:
	poetry run black ./**/*.py

.PHONY: sort
sort:
	poetry run isort ./**/*.py

.PHONY: test
test:
	poetry run pytest $(PACKAGE_DIR)

.PHONY: ass1-1
ass1-1:
	poetry run python $(ASS1)/task1.py