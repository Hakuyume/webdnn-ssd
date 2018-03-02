let WebDNN = require('webdnn');

let runner;
let file = document.getElementById("image");
let button = document.getElementById("button");

async function init() {
    file.disabled = true;
    button.disabled = true;

    runner = await WebDNN.load('./ssd300');

    file.disabled = false;
    button.disabled = false;
    button.onclick = run;
}

async function run() {
    try {
        file.disabled = true;
        button.disabled = true;

        let x = runner.getInputViews()[0];
        let mb_locs = runner.getOutputViews()[0];
        let mb_confs = runner.getOutputViews()[1];

        let options = {
            dstH: 300, dstW: 300,
            order: WebDNN.Image.Order.CHW,
            bias: [123, 117, 104]
        };
        let img = await WebDNN.Image.getImageArray(file, options);

        WebDNN.Image.setImageArrayToCanvas(
            img, 300, 300,
            document.getElementById('canvas'),
            options);

        x.set(img);
        await runner.run();
    } finally {
        file.disabled = false;
        button.disabled = false;
    }
}

init();
