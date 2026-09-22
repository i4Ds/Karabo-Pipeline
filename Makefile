# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
DOCSRC        = doc/src
SOURCEDIR     = _build
BUILDDIR      = _deploy

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Assemble the Sphinx source tree from doc/src. Done from scratch every time so
# that removed or renamed pages don't linger in $(SOURCEDIR).
prepare:
	python $(DOCSRC)/examples/combine_examples.py
	rm -rf "$(SOURCEDIR)"
	cp -a $(DOCSRC)/ "$(SOURCEDIR)"

clean:
	rm -rf "$(SOURCEDIR)" "$(BUILDDIR)"

.PHONY: help prepare clean Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: prepare Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
