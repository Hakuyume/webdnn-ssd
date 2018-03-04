# WebDNN-SSD
Single Shot MultiBox Detector on [WebDNN](https://mil-tokyo.github.io/webdnn/)

## [Demo](https://webdnn-ssd.netlify.com)

## Installation
Requirements

- Python 3
- [ChainerCV](https://github.com/chainer/chainercv)
- [WebDNN](https://mil-tokyo.github.io/webdnn/)
- nodejs/npm
- rust/cargo (nightly + wasm32-unknown-unknown)

```
$ pip install chainercv webdnn
$ npm install
$ rustup install nightly
$ rustup target add --toolchain nightly wasm32-unknown-unknown

$ make
$ cd dist/
$ python -m http.server
```
