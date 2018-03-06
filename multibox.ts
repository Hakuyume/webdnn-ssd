import {Box} from './nms';

export class ScoredBox implements Box {
    constructor(
        public y_min: number,
        public x_min: number,
        public y_max: number,
        public x_max: number,
        public score: number[]
    ) {}
}

function softmax(xs: number[]): number[] {
    const exp = xs.map((x) => Math.exp(x));
    const sum = exp.reduce((s, x) => s + x);
    return exp.map((e) => e / sum);
}

export class Multibox {
    constructor(
        public grids: number[],
        public aspect_ratios: number[][],
        public steps: number[],
        public sizes: number[],
        public variance: number[],
        public n_fg_class: number
    ) {}

    decode(xs: Float32Array[]): ScoredBox[] {
        const bbox: ScoredBox[] = [];

        for (let k = 0; k < this.grids.length; k++) {
            const loc = (d: number, c: number, v: number, u: number): number => {
                return xs[k * 2 + 0][((d * 4 + c) * this.grids[k] + v) * this.grids[k] + u];
            };
            const conf = (d: number, l: number, v: number, u: number): number => {
                return xs[k * 2 + 1][((d * (this.n_fg_class + 1) + l) * this.grids[k] + v) * this.grids[k] + u];
            };

            const default_bbox: number[][] = [];
            {
                const s = this.sizes[k];
                default_bbox.push([s, s]);
            }
            {
                const s = Math.sqrt(this.sizes[k] * this.sizes[k + 1]);
                default_bbox.push([s, s]);
            }
            {
                const s = this.sizes[k];
                for (const ar of this.aspect_ratios[k]) {
                    default_bbox.push([s / Math.sqrt(ar), s * Math.sqrt(ar)]);
                    default_bbox.push([s * Math.sqrt(ar), s / Math.sqrt(ar)]);
                }
            }

            for (const [d, default_bb] of default_bbox.entries()) {
                for (let v = 0; v < this.grids[k]; v++) {
                    for (let u = 0; u < this.grids[k]; u++) {
                        let cy = (v + 0.5) * this.steps[k];
                        let cx = (u + 0.5) * this.steps[k];
                        let [h, w] = default_bb;
                        cy += loc(d, 0, v, u) * this.variance[0] * h;
                        cx += loc(d, 1, v, u) * this.variance[0] * w;
                        h *= Math.exp(loc(d, 2, v, u) * this.variance[1]);
                        w *= Math.exp(loc(d, 3, v, u) * this.variance[1]);

                        let score: number[] = [];
                        for (let l = 0; l < this.n_fg_class + 1; l++) {
                            score.push(conf(d, l, v, u));
                        }
                        score = softmax(score);
                        score.shift();

                        bbox.push(new ScoredBox(
                            cy - h / 2, cx - w / 2,
                            cy + h / 2, cx + w / 2,
                            score
                        ));
                    }
                }
            }
        }

        return bbox;
    }
}
