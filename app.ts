import * as WebDNN from 'webdnn';
import {non_maximum_suppression} from './nms';
import {Multibox} from './multibox';

const html = {
    canvas: <HTMLCanvasElement>document.getElementById('canvas'),
    file: <HTMLInputElement>document.getElementById('image'),
    button: <HTMLInputElement>document.getElementById('button'),
    status: <HTMLElement>document.getElementById('status')
};

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

const multibox = new Multibox(
    [38, 19, 10, 5, 3, 1],
    [[2], [2, 3], [2, 3], [2, 3], [2,], [2,]],
    [8, 16, 32, 64, 100, 300],
    [30, 60, 111, 162, 213, 264, 315],
    [0.1, 0.2],
    label_names.length
);

let runner: WebDNN.DescriptorRunner | null = null;
let img: Float32Array | Int32Array | null = null;

async function init() {
    html.file.onchange = async () => {
        const options = {
            dstH: 300, dstW: 300,
            order: WebDNN.Image.Order.CHW,
            bias: [123, 117, 104]
        };
        img = await WebDNN.Image.getImageArray(html.file, options);
        WebDNN.Image.setImageArrayToCanvas(img, 300, 300, html.canvas, options);
        html.button.disabled = false;
        html.status.textContent = '';
    };
    html.button.onclick = run;
}

async function run() {
    try {
        html.file.disabled = true;
        html.button.disabled = true;

        if (runner == null) {
            html.status.textContent = 'Loading ...';
            runner = await WebDNN.load(
                './model',
                {progressCallback: (loaded, total) => html.status.textContent = `Loading ... ${(loaded / total * 100).toFixed(1)}%`});
        }

        if (img == null) {
            throw 'Null Image';
        }
        runner.getInputViews()[0].set(img);
        const outputs = runner.getOutputViews();

        html.status.textContent = 'Computing ...';
        await runner.run();
        const bbox = multibox.decode(outputs.map((v) => v.toActual()));

        html.status.textContent = 'Visualizing ...';
        const ctx = html.canvas.getContext('2d');
        if (ctx == null) {
            throw 'Null Context';
        }

        for (let l = 0; l < label_names.length; l++) {
            let bbox_l = bbox.filter((bb) => bb.score[l] >= 0.6);
            bbox_l.sort((bb0, bb1) => bb1.score[l] - bb0.score[l]);
            bbox_l = non_maximum_suppression(bbox_l, 0.45);

            for (const bb of bbox_l) {
                ctx.fillText(label_names[l], bb.x_min, bb.y_min);
                ctx.beginPath();
                ctx.strokeStyle = 'red';
                ctx.strokeRect(bb.x_min, bb.y_min, bb.x_max, bb.y_max);
            }
        }

        html.status.textContent = 'Done';
    } catch (err) {
        html.status.textContent = 'Error';
        throw err;
    } finally {
        html.file.disabled = false;
        html.button.disabled = false;
    }
}

init();
