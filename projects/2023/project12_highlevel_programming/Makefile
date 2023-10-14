.PHONY: report.pdf all clean

# The first rule in a Makefile is the one executed by default ("make"). It
# should always be the "all" rule, so that "make" and "make all" are identical.
all: report.pdf

%.tex: %.raw
	./raw2tex $< > $@

%.tex: %.dat
	./dat2tex $< > $@

report.pdf: report.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make report.tex

clean:
	latexmk -CA
