.PHONY: install install-all test compile check-data run
install:
	python -m pip install -e .
install-all:
	python -m pip install -e '.[all,dev]'
test:
	python -m pytest -q
compile:
	python -m compileall -q src main.py
check-data:
	python scripts/check_data.py
run:
	python main.py
