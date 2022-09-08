.PHONY: all install lint test format

all: lint test

install:
	pip install -r requirements.txt  -e .

lint:
	pydocstyle .
	flake8 .
	isort .
	mypy --ignore-missing-imports --no-strict-optional --disable-error-code call-overload --disable-error-code arg-type \
	--disable-error-code attr-defined --disable-error-code assignment \
	--disable-error-code operator --disable-error-code index \
	--show-error-codes .
	pylama .
	pylint --disable W1514 *.py


test:
	pytest tests

format:
	isort .
	# black .
