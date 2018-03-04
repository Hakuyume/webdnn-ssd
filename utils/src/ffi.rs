use std::cmp::Ordering;
use std::mem;
use std::slice;

use super::Rect;
use super::non_maximum_suppression_by;

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub unsafe extern "C" fn malloc(len: usize) -> *mut f32 {
    let mut buf = Vec::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    mem::forget(buf);
    ptr
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub unsafe extern "C" fn free(ptr: *mut ()) {
    Box::from_raw(ptr);
}

#[no_mangle]
pub unsafe extern "C" fn non_maximum_suppression(n_bbox: usize,
                                                 bbox: *const f32,
                                                 bbox_stride: usize,
                                                 score: *const f32,
                                                 score_stride: usize,
                                                 nms_thresh: f32,
                                                 score_thresh: f32)
                                                 -> *mut usize {
    let score = slice::from_raw_parts(score, (n_bbox - 1) * score_stride + 1);
    let sc = |i| score[i * score_stride];
    let mut indices: Vec<_> = (0..n_bbox).filter(|&i| sc(i) >= score_thresh).collect();
    indices.sort_unstable_by(|&i, &j| sc(j).partial_cmp(&sc(i)).unwrap_or(Ordering::Equal));

    #[derive(Clone, Copy)]
    struct Bb<'a>(&'a [f32]);
    impl<'a> Rect<f32> for Bb<'a> {
        fn x_min(&self) -> f32 {
            self.0[1]
        }
        fn y_min(&self) -> f32 {
            self.0[0]
        }
        fn x_max(&self) -> f32 {
            self.0[3]
        }
        fn y_max(&self) -> f32 {
            self.0[2]
        }
    }
    let bbox = slice::from_raw_parts(bbox, (n_bbox - 1) * bbox_stride + 4);
    let bb = |i| Bb(&bbox[i * bbox_stride..i * bbox_stride + 4]);
    let mut indices: Vec<_> =
        non_maximum_suppression_by(indices.into_iter(), |&i| bb(i), nms_thresh).collect();

    indices.push(n_bbox);
    let ptr = indices.as_mut_ptr();
    mem::forget(indices);
    ptr
}
