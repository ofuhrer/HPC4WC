# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
AUTODOC_SOURCEDIR = ../src/icon4py/
AUTODOC_OPTS = --implicit-namespaces
AUTODOC_DIR = _source
# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: autodoc clean help Makefile
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(AUTODOC_DIR)/*

autodoc:
	@echo
	@echo "Running sphinx-apidoc..."
	sphinx-apidoc ${AUTODOC_OPTS} -o ${AUTODOC_DIR} ${AUTODOC_SOURCEDIR}
	@echo
	@echo "sphinx-apidoc finished. The generated autodocs are in $(AUTODOC_DIR)."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: autodoc Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
