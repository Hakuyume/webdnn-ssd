.PHONY: all
all: dist/index.html dist/model dist/bundle.js dist/utils.wasm

dist:
	mkdir dist

dist/index.html: dist index.html
	cp index.html dist/

dist/model: dist convert.py
	python convert.py --out dist/model

dist/bundle.js: dist js/main.js
	cd js; npm run build
	mv js/bundle.js dist/

dist/utils.wasm: dist utils/src/*
	cd utils; cargo build --release --target wasm32-unknown-unknown
	mv utils/target/wasm32-unknown-unknown/release/utils.wasm dist/
