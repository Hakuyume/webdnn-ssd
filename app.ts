import * as WebDNN from 'webdnn';

async function load_wasm(url: string) {
    const response = await fetch(url);
    const bytes = await response.arrayBuffer();
    const result = await WebAssembly.instantiate(bytes);
    return result.instance.exports;
}

const html = {
    canvas: <HTMLCanvasElement>document.getElementById('canvas'),
    file: <HTMLInputElement>document.getElementById('image'),
    button: <HTMLInputElement>document.getElementById('button'),
    status: <HTMLElement>document.getElementById('status')};

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

type usize = number;
type usize_ptr = number;
type f32 = number;
type f32_ptr = number;
type Utils = {
    memory: WebAssembly.Memory,
    malloc: (len: usize) => f32_ptr,
    free: (ptr: f32_ptr) => void,
    non_maximum_suppression: (
        n_bbox: usize,
        bbox: f32_ptr,
        bbox_stride: usize,
        score: f32_ptr,
        score_stride: usize,
        nms_thresh: f32,
        score_thresh: f32) => usize_ptr
}

let utils: Utils | null = null;
let runner: WebDNN.DescriptorRunner | null = null;
let img: Float32Array | Int32Array | null = null;

async function init() {
    utils = await load_wasm('./utils.wasm');
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
        if (utils == null) {
            throw 'Null Utils';
        }

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

        html.status.textContent = 'Visualizing ...';

        const n_bbox_k = [5776, 2166, 600, 150, 36, 4];
        const n_bbox = n_bbox_k.reduce((s, x) => s + x);
        const n_class = label_names.length;

        const bbox_ptr = utils.malloc(n_bbox * 4);
        const score_ptr = utils.malloc(n_bbox * n_class);

        for (let k = 0, offset = 0; k < n_bbox_k.length; k++, offset += n_bbox_k[k]) {
            new Float32Array(utils.memory.buffer, bbox_ptr)
                .set(outputs[k * 2 + 0].toActual(), offset * 4);
            new Float32Array(utils.memory.buffer, score_ptr)
                .set(outputs[k * 2 + 1].toActual(), offset * n_class);
        }

        const ctx = html.canvas.getContext('2d');
        if (ctx == null) {
            throw 'Null Context';
        }

        for (let lb = 0; lb < n_class; lb++) {
            const indices_ptr = utils.non_maximum_suppression(
                n_bbox,
                bbox_ptr, 4,
                score_ptr + lb * 4, n_class,
                0.45, 0.6);

            const indices = new Uint32Array(utils.memory.buffer, indices_ptr);
            const bbox = new Float32Array(utils.memory.buffer, bbox_ptr);

            for (let k = 0; indices[k] < n_bbox; k++) {
                const [t, l, b, r] = bbox.slice(indices[k] * 4, (indices[k] + 1) * 4);

                ctx.fillText(label_names[lb], l, t);
                ctx.beginPath();
                ctx.strokeStyle = 'red';
                ctx.strokeRect(l, t, r, b);
            }

            utils.free(indices_ptr);
        }

        utils.free(bbox_ptr);
        utils.free(score_ptr);

        html.status.textContent = 'Done';
    } catch (err) {
        html.status.textContent = 'Error';
    } finally {
        html.file.disabled = false;
        html.button.disabled = false;
    }
}

init();
