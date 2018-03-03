.PHONY: all
all: index.html bundle.js utils.wasm model

bundle.js: js/main.js
	cd js; npm run build
	cp js/bundle.js .

utils.wasm: utils/src/*
	cd utils; cargo build --release --target wasm32-unknown-unknown
	cp utils/target/wasm32-unknown-unknown/release/utils.wasm .

model:
	python convert.py
