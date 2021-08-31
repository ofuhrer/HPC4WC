.DEFAULT_GOAL := run

.PHONY: run
run:
	PYTHONPATH=$$PYTHONPATH:$$PWD python3 example/basic_function.py

.PHONY: clean_all
clean_all:
	rm -rf .codecache

.PHONY: clean_cpp
clean_cpp:
	rm -rf .codecache/cpp_*

.PHONY: clean_python
clean_python:
	rm -rf .codecache/generated_*.py
	rm -rf .codecache/__pycache__
