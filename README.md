# Setup

```
uv sync
uv tool install ./external/bril-txt # adds bril2json and bril2txt
make -C ./external/brilift install # adds brilift
deno install -g ./external/brili.ts # adds brili
deno install -g --allow-env --allow-read ./external/ts2bril.ts # adds ts2bril
```