use std::mem;
use std::slice;

use super::bbox;

unsafe fn alloc<T>(len: usize) -> *mut T {
    let mut buf = Vec::<T>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    mem::forget(buf);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn alloc_f32(len: usize) -> *mut f32 {
    alloc(len)
}

#[no_mangle]
pub unsafe fn free(ptr: *mut ()) {
    Box::from_raw(ptr);
}

#[no_mangle]
pub unsafe extern "C" fn non_maximum_suppression(n_bbox: usize,
                                                 bbox: *const bbox::Bbox,
                                                 bbox_stride: usize,
                                                 score: *const f32,
                                                 score_stride: usize,
                                                 nms_thresh: f32,
                                                 score_thresh: f32)
                                                 -> *mut usize {
    let bbox = slice::from_raw_parts(bbox, (n_bbox - 1) * bbox_stride + 1);
    let score = slice::from_raw_parts(score, (n_bbox - 1) * score_stride + 1);

    let mut indices = bbox::non_maximum_suppression(n_bbox,
                                                    |i| &bbox[i * bbox_stride],
                                                    |i| score[i * score_stride],
                                                    nms_thresh,
                                                    score_thresh);

    indices.push(n_bbox);
    let ptr = indices.as_mut_ptr();
    mem::forget(indices);
    ptr
}
