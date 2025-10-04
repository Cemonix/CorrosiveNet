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
pub mod broadcast;
pub mod traits;

pub use core::TensorCore;
pub use traits::{TensorElement, TensorNum, TensorSigned, TensorFloat, TensorBool};
pub use storage::TensorStorage;
pub use dim::TensorDims;
pub use scalar::TensorScalar;
pub use shape::TensorShape;
pub use stats::TensorStats;
pub use mask::TensorMask;
pub use linalg::TensorLinAlg;
pub use arithmetic::TensorArithmetic;
pub use math::TensorMath;
pub use comparison::TensorComparison;
pub use broadcast::TensorBroadcast;

#[derive(Debug, Clone)]
pub enum Device {
    CPU,
    CUDA(usize), // Device index for CUDA
}

impl Default for Device {
    fn default() -> Self {
        Device::CPU
    }
}

impl Device {
    /// Create a CUDA device with default GPU index (0)
    pub fn cuda() -> Self {
        Device::CUDA(0)
    }
}

#[derive(Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    device: Device
}

impl<T> Tensor<T> {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn has_same_device(&self, other: &Tensor<T>) -> bool {
        match (&self.device, &other.device) {
            (Device::CPU, Device::CPU) => true,
            (Device::CUDA(idx1), Device::CUDA(idx2)) => idx1 == idx2,
            _ => false,
        }
    }
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
