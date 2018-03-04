#![feature(conservative_impl_trait)]

extern crate num_traits;

mod rect;
pub use rect::*;

mod nms;
pub use nms::*;

pub mod ffi;
