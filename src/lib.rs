#![deny(unused_crate_dependencies)]
#![warn(clippy::all)]

pub mod field;
pub mod key;
pub mod linalg;

pub mod hazmat {
    pub mod expand;
    pub mod hash;
}
