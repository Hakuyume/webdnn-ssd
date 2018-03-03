use std::cmp;
use std::mem;
use std::slice;

#[repr(C)]
pub struct Bbox {
    y_min: f32,
    x_min: f32,
    y_max: f32,
    x_max: f32,
}

#[no_mangle]
pub unsafe extern "C" fn non_maximum_suppression(n_bbox: usize,
                                                 mb_bbox: *const Bbox,
                                                 mb_score: *const f32,
                                                 score_offset: usize,
                                                 score_stride: usize)
                                                 -> *mut i32 {
    let mb_bbox = slice::from_raw_parts(mb_bbox, n_bbox);
    let mb_score = slice::from_raw_parts(mb_score, n_bbox * score_stride);


    let mut posi: Vec<_> = (0..n_bbox)
        .map(|i| (i, mb_score[score_offset + i * score_stride]))
        .filter(|&(_, s)| s >= 0.6)
        .collect();
    posi.sort_unstable_by(|&(_, s0), &(_, s1)| s1.partial_cmp(&s0).unwrap_or(cmp::Ordering::Equal));

    let mut indices = Vec::new();
    for (i, _) in posi.into_iter() {
        if indices
               .iter()
               .all(|&j| mb_bbox[i].iou(&mb_bbox[j as usize]) < 0.45) {
            indices.push(i as i32);
        }
    }

    indices.push(-1);
    let ptr = indices.as_mut_ptr();
    mem::forget(indices);
    ptr
}

impl Bbox {
    fn area(&self) -> f32 {
        if self.y_min < self.y_max && self.x_min < self.x_max {
            (self.y_max - self.y_min) * (self.x_max - self.x_min)
        } else {
            0.
        }
    }

    fn iou(&self, other: &Self) -> f32 {
        let u = Bbox {
            y_min: self.y_min.max(other.y_min),
            x_min: self.x_min.max(other.x_min),
            y_max: self.y_max.min(other.y_max),
            x_max: self.x_max.min(other.x_max),
        };
        u.area() / (self.area() + other.area() - u.area())
    }
}
