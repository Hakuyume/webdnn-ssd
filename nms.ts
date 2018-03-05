interface Bb {
    y_min: number;
    x_min: number;
    y_max: number;
    x_max: number;
}

function area(bb: Bb): number {
    if (bb.y_min < bb.y_max && bb.x_min < bb.x_max) {
        return (bb.y_max - bb.y_min) * (bb.x_max - bb.x_min);
    } else {
        return 0;
    }
}

function iou(bb0: Bb, bb1: Bb): number {
    const inter = area({
        y_min: Math.max(bb0.y_min, bb1.y_min),
        x_min: Math.max(bb0.x_min, bb1.x_min),
        y_max: Math.min(bb0.y_max, bb1.y_max),
        x_max: Math.min(bb0.x_max, bb1.x_max)
    });
    return inter / (area(bb0) + area(bb1) - inter);
}

export function non_maximum_suppression(bbox: Bb[], thresh: number): Bb[] {
    let selected_bbox: Bb[] = [];
    for (let bb of bbox) {
        if (selected_bbox.every((selected_bb) => iou(bb, selected_bb) < thresh)) {
            selected_bbox.push(bb);
        }
    }
    return selected_bbox;
}
