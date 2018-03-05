import * as WebDNN from 'webdnn';
import {non_maximum_suppression} from './nms';

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

let runner: WebDNN.DescriptorRunner | null = null;
let img: Float32Array | Int32Array | null = null;

async function init() {
    html.file.oninput = async () => {
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
        const output_views = runner.getOutputViews();

        html.status.textContent = 'Computing ...';
        await runner.run();
        const outputs = output_views.map((v) => v.toActual());

        html.status.textContent = 'Visualizing ...';

        const ctx = html.canvas.getContext('2d');
        if (ctx == null) {
            throw 'Null Context';
        }

        const n_class = label_names.length;
        for (let lb = 0; lb < n_class; lb++) {
            let bbox = [];
            for (let [k, n] of [5776, 2166, 600, 150, 36, 4].entries()) {
                for (let i = 0; i < n; i++) {
                    bbox.push({
                        y_min: outputs[k * 2 + 0][i * 4 + 0],
                        x_min: outputs[k * 2 + 0][i * 4 + 1],
                        y_max: outputs[k * 2 + 0][i * 4 + 2],
                        x_max: outputs[k * 2 + 0][i * 4 + 3],
                        score: outputs[k * 2 + 1][i * n_class + lb]
                    });
                }
            }

            bbox = bbox.filter((bb) => bb.score >= 0.6);
            bbox.sort((bb0, bb1) => bb1.score - bb0.score);
            bbox = non_maximum_suppression(bbox, 0.45);

            for (let bb of bbox) {
                ctx.fillText(label_names[lb], bb.x_min, bb.y_min);
                ctx.beginPath();
                ctx.strokeStyle = 'red';
                ctx.strokeRect(bb.x_min, bb.y_min, bb.x_max, bb.y_max);
            }
        }

        html.status.textContent = 'Done';
    } catch (err) {
        html.status.textContent = 'Error';
    } finally {
        html.file.disabled = false;
        html.button.disabled = false;
    }
}

init();
