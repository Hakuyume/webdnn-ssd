'use strict';

const WebDNN = require('webdnn');
let runner;
let utils;

const canvas = document.getElementById('canvas');
const file = document.getElementById("image");
const button = document.getElementById("button");

async function load_wasm(url) {
    const response = await fetch(url);
    const bytes = await response.arrayBuffer();
    const result = await WebAssembly.instantiate(bytes);
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

        const options = {
            dstH: 300, dstW: 300,
            order: WebDNN.Image.Order.CHW,
            bias: [123, 117, 104]
        };
        const img = await WebDNN.Image.getImageArray(file, options);

        WebDNN.Image.setImageArrayToCanvas(img, 300, 300, canvas, options);

        runner.getInputViews()[0].set(img);
        const outputs = runner.getOutputViews();

        await runner.run();

        const n_bbox_k = [5776, 2166, 600, 150, 36, 4];
        const n_bbox = n_bbox_k.reduce((s, x) => s + x);
        const n_class = label_names.length;

        const bbox_ptr = utils.alloc_f32(n_bbox * 4);
        const score_ptr = utils.alloc_f32(n_bbox * n_class);

        for (let k = 0, offset = 0; k < n_bbox_k.length; k++, offset += n_bbox_k[k]) {
            new Float32Array(utils.memory.buffer, bbox_ptr)
                .set(outputs[k * 2 + 0].toActual(), offset * 4);
            new Float32Array(utils.memory.buffer, score_ptr)
                .set(outputs[k * 2 + 1].toActual(), offset * n_class);
        }

        const ctx = canvas.getContext('2d');

        for (let lb = 0; lb < n_class; lb++) {
            const indices_ptr = utils.non_maximum_suppression(
                n_bbox,
                bbox_ptr, 1,
                score_ptr + lb * 4, n_class,
                0.45, 0.6);
            const indices = new Uint32Array(utils.memory.buffer, indices_ptr);

            for (let k = 0; indices[k] < n_bbox; k++) {
                const i = indices[k];
                const bbox = new Float32Array(utils.memory.buffer, bbox_ptr);
                const t = bbox[i * 4 + 0];
                const l = bbox[i * 4 + 1];
                const b = bbox[i * 4 + 2];
                const r = bbox[i * 4 + 3];

                ctx.fillText(label_names[lb], l, t);
                ctx.beginPath();
                ctx.strokeStyle = 'red';
                ctx.strokeRect(l, t, r, b);
            }

            utils.free(indices_ptr);
        }

        utils.free(bbox_ptr);
        utils.free(score_ptr);
    } finally {
        file.disabled = false;
        button.disabled = false;
    }
}

init();
