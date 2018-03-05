.PHONY: all
all: dist/index.html dist/model dist/bundle.js dist/utils.wasm

dist/index.html: index.html
	mkdir -p dist
	cp index.html dist/

dist/model: convert.py
	- rm -r dist/model
ifdef EIGEN
	python convert.py --out dist/model --backend webgl webassembly --eigen $(EIGEN)
else
	python convert.py --out dist/model --backend webgl
endif

dist/bundle.js: app.ts
	npm run build

dist/utils.wasm: utils/src/*
	cd utils; rustup run nightly cargo build --release --target wasm32-unknown-unknown
	mkdir -p dist
	- mv utils/target/wasm32-unknown-unknown/release/utils.wasm dist/
