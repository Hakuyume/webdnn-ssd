use std::mem;

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
