# WebDNN-SSD
Single Shot MultiBox Detector on [WebDNN](https://mil-tokyo.github.io/webdnn/)

## [Demo](https://webdnn-ssd.netlify.com)

## Installation
Requirements

- [ChainerCV](https://github.com/chainer/chainercv)
- [WebDNN](https://mil-tokyo.github.io/webdnn/)
- nodejs/npm
- rust/cargo (wasm32-unknown-unknown)

```
$ pip install chainercv webdnn
$ cd js/; npm install; cd -
$ make
$ cd dist/
$ python -m http.server
```
