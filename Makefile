PROG ?= example.bril

OPTS = uv run python -m src.env.opt.tdce tdce+ | uv run python -m src.env.opt.lvn -p -c -f

.PHONY: all interpret optimize compile measure

all: interpret

interpret:
	bril2json < $(PROG) | brili -p

transpile:
	ts2bril $(PROG) | bril2txt > $(PROG:.ts=.bril)

optimize:
	bril2json < $(PROG) | $(OPTS) | bril2txt > $(PROG:.bril=.optimized.bril)

compile:
	bril2json < $(PROG) | brilift -O none -o output.o

measure:
	@TMP_DIR=$$(mktemp -d); \
	trap 'rm -rf "$$TMP_DIR"' EXIT; \
	bril2json < $(PROG) | brilift -O none -o "$$TMP_DIR/base.o"; \
	bril2json < $(PROG) | $(OPTS) | brilift -O none -o "$$TMP_DIR/opt.o"; \
	BASE=$$(wc -c < "$$TMP_DIR/base.o" | tr -d ' '); \
	OPT=$$(wc -c < "$$TMP_DIR/opt.o" | tr -d ' '); \
	DIFF=$$((BASE - OPT)); \
	PERCENT=$$((DIFF * 100 / BASE)); \
	echo "-----------------------------------------"; \
	echo " Plik: $(PROG)"; \
	echo "-----------------------------------------"; \
	echo " Rozmiar bazowy:          $$BASE bajtów"; \
	echo " Rozmiar po optymalizacji: $$OPT bajtów"; \
	echo " Zysk (redukcja):          $$DIFF bajtów ($$PERCENT%)"; \
	echo "-----------------------------------------"