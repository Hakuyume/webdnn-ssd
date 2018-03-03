# WebDNN-SSD
Single Shot MultiBox Detector on [WebDNN](https://mil-tokyo.github.io/webdnn/)

## [Demo](https://hakuyume.github.io/webdnn-ssd/)

## Known Issues
- Only WebAssembly backend works
    - WebGL: stuck on `webdnn.backend.generate_descriptor`
    - fallback: some operations are not supported by WebDNN.
- Wrong result
    - The result is different from that of [ChainerCV](https://github.com/chainer/chainercv)

## Installation
### 1. Convert model from ChainerCV to WebDNN (`model/`)
```
$ pip install chainercv webdnn
$ python convert.py
```
Currenly, `convert.py` uses WebAssembly backend.
Therefore, it requires `emscripten` and `Eigen`.
You can specify the include path of `Eigen` by `--eigen` option.

### 2. Transpile JavaScript (`bundle.js`)
```
$ cd js/
$ npm install
$ npm run build
$ cp bundle.js ../
```

### 3. Compile WebAssembly utility (`utils.wasm`)
```
$ cd utils/
$ cargo build --release --target wasm32-unknown-unknown
$ cp target/wasm32-unknown-unknown/release/utils.wasm ../
```

### 4. Launch a web server
Please make sure the working directly contains following files/directories.
```
./
./index.html
./model/
./bundle.js
./utils.wasm
```

You can use `http.server` for testing.
```
$ python -m http.server
```
