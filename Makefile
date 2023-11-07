.PHONY: all install lint test format

all: lint test

install:
	pip install -r requirements.txt

lint:
	isort src
	flake8 --max-line-length=100 src
	pylama --ignore=E501 src
	pylint src
	mypy --ignore-missing-imports --no-strict-optional --disable-error-code call-overload --disable-error-code arg-type \
	--disable-error-code attr-defined --disable-error-code assignment \
	--disable-error-code operator --disable-error-code index --disable-error-code misc \
	--show-error-codes src

test:
	pytest tests
