#![feature(conservative_impl_trait)]
#![feature(try_from)]

extern crate num_traits;

mod rect;
pub use rect::*;

mod nms;
pub use nms::*;

pub mod ffi;
