PYTHON_VERSION ?= 3.8.0
TAG_NAME ?= ofuhrer/shallow_water:latest
WORK_DIR ?= /work

help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## build Docker container image
	docker build --build-arg PYVERSION=$(PYTHON_VERSION) -t $(TAG_NAME) .

enter: ## run Docker container and enter using volume mount for development
	docker run -it -v $(shell pwd):$(WORK_DIR) -w $(WORK_DIR) --rm $(TAG_NAME)

