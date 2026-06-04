.PHONY: all watch clean

TEXFILES := $(wildcard *.tex)
PDFFILES := $(TEXFILES:.tex=.pdf)
BUILDDIR := _build

all: $(PDFFILES)

%.pdf: %.tex
	latexmk -quiet -bibtex -f -pdfxe -use-make -outdir=$(BUILDDIR) $(PREVIEW_CONTINUOUSLY) ./$<
	cp $(BUILDDIR)/$*.pdf ./$@

watch: PREVIEW_CONTINUOUSLY=-pvc
watch: $(PDFFILES)

clean:
	latexmk -c
	rm -f *.bbl
	rm -f *.synctex.gz
	rm -rf $(BUILDDIR)
	rm -f $(PDFFILES)