SHELL := /bin/sh

MAIN := report
IMAGE := ghcr.io/xu-cheng/texlive-full:latest

# Rootless podman build, for the arm64 cluster (santis): $(IMAGE) above is
# amd64-only, and podman's default overlay store lives on the network home
# filesystem, which does not support the xattrs it needs. So use a multi-arch
# image and a vfs store on node-local /tmp. Note /tmp is tmpfs, so the image is
# gone after a reboot or on a different node and is re-pulled on the next run.
PODMAN_IMAGE := docker.io/texlive/texlive:latest-medium
PODMAN_STORE := /tmp/$(USER)-podman-texlive
PODMAN := podman --root $(PODMAN_STORE)/root --runroot $(PODMAN_STORE)/runroot \
	--storage-driver vfs

.PHONY: all watch clean docker docker-clean podman podman-clean

all:
	latexmk $(MAIN).tex

watch:
	latexmk -pvc $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex
	rm -rf build

docker:
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$$PWD:/work" -w /work $(IMAGE) \
		latexmk $(MAIN).tex

docker-clean:
	docker run --rm --user "$$(id -u):$$(id -g)" \
		-v "$$PWD:/work" -w /work $(IMAGE) \
		sh -c 'latexmk -C $(MAIN).tex; rm -rf build'

podman:
	mkdir -p $(PODMAN_STORE)/root $(PODMAN_STORE)/runroot
	$(PODMAN) run --rm -v "$$PWD:/work:Z" -w /work $(PODMAN_IMAGE) \
		latexmk $(MAIN).tex

podman-clean:
	rm -rf build $(MAIN).pdf
	rm -rf $(PODMAN_STORE)
