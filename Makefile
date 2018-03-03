.PHONY: all
all: dist/index.html dist/model dist/bundle.js dist/utils.wasm

dist/index.html: index.html
	mkdir -p dist
	cp index.html dist/

dist/model: convert.py
	mkdir -p dist
	python convert.py --out dist/model --backend webgl
ifdef EIGEN
	python convert.py --out dist/model --backend webassembly --eigen $(EIGEN)
endif

dist/bundle.js: js/main.js
	cd js; npm run build
	mkdir -p dist
	mv js/bundle.js dist/

dist/utils.wasm: utils/src/*
	cd utils; cargo build --release --target wasm32-unknown-unknown
	cd js; npm run build
	mv utils/target/wasm32-unknown-unknown/release/utils.wasm dist/
