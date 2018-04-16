.PHONY: all
all: dist/index.html dist/model dist/bundle.js

dist/index.html: index.html
	mkdir -p dist
	cp index.html dist/

dist/model: convert.py
	- rm -r dist/model
	python convert.py --out dist/model --backend webgl webassembly

dist/bundle.js: *.ts
	npm run build
