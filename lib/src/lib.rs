//!
//! This crate provides [`ArrayStump`](./struct.ArrayStump.html), a data structure mixing dynamic array and sorted set semantics.
//!
//! For algorithmic notes see: [README on GitHub](https://github.com/bluenote10/rust-array-stump)
//!
//! # Example
//!
//! ```
//! use array_stump::ArrayStump;
//!
//! fn comparator(a: &i32, b: &i32) -> std::cmp::Ordering {
//!     a.cmp(b)
//! }
//!
//! let mut array_stump = ArrayStump::new(comparator);
//!
//! array_stump.insert(2);
//! array_stump.insert(3);
//! array_stump.insert(1);
//!
//! array_stump.remove(&2);
//!
//! assert_eq!(array_stump.collect(), vec![1, 3]);
//! ```
//!

mod array_stump;

pub use crate::array_stump::ArrayStump;
