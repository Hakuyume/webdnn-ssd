'use strict';

let WebDNN = require('webdnn');
let runner;
let utils;

let canvas = document.getElementById('canvas');
let file = document.getElementById("image");
let button = document.getElementById("button");

async function load_wasm(url) {
    let response = await fetch(url);
    let bytes = await response.arrayBuffer();
    let result = await WebAssembly.instantiate(bytes);
    return result.instance.exports;
}

const label_names = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'];


async function init() {
    file.disabled = true;
    button.disabled = true;

    runner = await WebDNN.load('./model');
    utils = await load_wasm('./utils.wasm');

    file.disabled = false;
    button.disabled = false;
    button.onclick = run;
}

async function run() {
    try {
        file.disabled = true;
        button.disabled = true;

        let options = {
            dstH: 300, dstW: 300,
            order: WebDNN.Image.Order.CHW,
            bias: [123, 117, 104]
        };
        let img = await WebDNN.Image.getImageArray(file, options);

        WebDNN.Image.setImageArrayToCanvas(img, 300, 300, canvas, options);

        runner.getInputViews()[0].set(img);
        await runner.run();

        let mb_bbox = runner.getOutputViews()[0].toActual();
        let mb_score = runner.getOutputViews()[1].toActual();

        let mb_bbox_ptr = utils.alloc_f32(mb_bbox.length);
        let mb_score_ptr = utils.alloc_f32(mb_score.length);
        new Float32Array(utils.memory.buffer, mb_bbox_ptr)
            .set(mb_bbox);
        new Float32Array(utils.memory.buffer, mb_score_ptr)
            .set(mb_score);

        let n_bbox = mb_bbox.length / 4;
        let n_fg_class = mb_score.length * 4 / mb_bbox.length;
        let ctx = canvas.getContext('2d');

        for (let lb = 0; lb < n_fg_class; lb++) {
            let indices_ptr = utils.non_maximum_suppression(
                n_bbox,
                mb_bbox_ptr, 1,
                mb_score_ptr + lb * 4, n_fg_class,
                0.45, 0.6);
            let indices = new Uint32Array(utils.memory.buffer, indices_ptr);

            for (let k = 0; indices[k] < n_bbox; k++) {
                let i = indices[k];
                let t = mb_bbox[i * 4 + 0];
                let l = mb_bbox[i * 4 + 1];
                let b = mb_bbox[i * 4 + 2];
                let r = mb_bbox[i * 4 + 3];

                ctx.fillText(label_names[lb], l, t);
                ctx.beginPath();
                ctx.strokeStyle = 'red';
                ctx.strokeRect(l, t, r, b);
            }

            utils.free(indices_ptr);
        }

        utils.free(mb_bbox_ptr);
        utils.free(mb_score_ptr);
    } finally {
        file.disabled = false;
        button.disabled = false;
    }
}

init();
