pub mod core;
pub mod storage;
pub mod display;
pub mod dim;
pub mod shape;
pub mod scalar;
pub mod stats;
pub mod mask;
pub mod linalg;
pub mod arithmetic;
pub mod math;
pub mod comparison;
pub mod traits;

pub use core::TensorCore;
pub use traits::{TensorElement, TensorNum, TensorSigned, TensorFloat, TensorBool};
pub use storage::TensorStorage;
pub use dim::TensorDims;
pub use scalar::TensorScalar;
pub use shape::TensorShape;
pub use stats::TensorStats;
pub use mask::TensorMask;
pub use linalg::TensorLinearAlgebra;
pub use arithmetic::TensorArithmetic;
pub use math::TensorMath;
pub use comparison::TensorComparison;

#[derive(Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[derive(Debug)]
pub struct TensorError {
    details: String,
}

impl TensorError {
    pub fn new(details: &str) -> Self {
        TensorError {
            details: details.to_string(),
        }
    }
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tensor error: {}", self.details)
    }
}
