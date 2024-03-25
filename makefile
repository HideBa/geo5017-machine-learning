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

.PHONY: ass1-2a
ass1-2a:
	poetry run python $(ASS1)/task2a.py

.PHONY: ass1-2b
ass1-2b:
	poetry run python $(ASS1)/task2b.py

.PHONY: ass1-2c
ass1-2c:
	poetry run python $(ASS1)/task2c.py

.PHONY: ass2-vis
ass2-vis:
	poetry run python $(ASS2)/visualize.py


.PHONEY: ass2-all
ass2-all:
	poetry run python $(ASS2)/main.py