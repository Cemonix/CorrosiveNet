#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice};

pub mod core;
pub mod init;
pub mod display;
pub mod dim;
pub mod shape;
pub mod scalar;
pub mod stats;
pub mod logical;
pub mod linalg;
pub mod arithmetic;
pub mod math;
pub mod comparison;
pub mod broadcast;
pub mod traits;

pub use core::TensorCore;
pub use traits::{TensorElement, TensorNum, TensorSigned, TensorFloat, TensorBool};
pub use init::TensorInit;
pub use dim::TensorDims;
pub use scalar::TensorScalar;
pub use shape::TensorShape;
pub use stats::TensorStats;
pub use logical::TensorMask;
pub use linalg::TensorLinAlg;
pub use arithmetic::TensorArithmetic;
pub use math::TensorMath;
pub use comparison::TensorComparison;
pub use broadcast::TensorBroadcast;

/// Represents the device where tensor data is stored
#[derive(Debug, Clone, PartialEq, Eq)]
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
    #[cfg(feature = "cuda")]
    pub fn cuda() -> Self {
        Device::CUDA(0)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn cuda() -> Self {
        panic!("CUDA support not compiled. Rebuild with --features cuda");
    }
}

/// Internal storage for tensor data - either on CPU or GPU
#[derive(Clone)]
pub enum TensorStorage<T> {
    CPU(Vec<T>),
    #[cfg(feature = "cuda")]
    CUDA {
        context: Arc<CudaContext>,
        buffer: CudaSlice<T>,
        device_idx: usize,
    }
}

#[derive(Clone)]
pub struct Tensor<T> {
    storage: TensorStorage<T>,
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
