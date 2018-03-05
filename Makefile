.PHONY: all
all: dist/index.html dist/model dist/bundle.js

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

dist/bundle.js: *.ts
	npm run build
