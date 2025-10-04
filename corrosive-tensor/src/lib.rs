mod tensor;

#[cfg(feature = "cuda")]
mod cuda;

// Core types always available at root
pub use tensor::{Tensor, TensorError, Device};

// Traits available for explicit import
pub use tensor::{
    TensorElement, TensorNum, TensorSigned, TensorFloat, TensorBool,
    TensorCore, TensorInit, TensorDims, TensorScalar, TensorShape,
    TensorStats, TensorMask, TensorLinAlg, TensorArithmetic, TensorMath,
    TensorComparison, TensorBroadcast,
};

// CUDA backend
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

// Prelude for convenience
pub mod prelude {
    pub use crate::tensor::{
        Tensor, TensorError, Device,
        TensorElement, TensorNum, TensorSigned, TensorFloat, TensorBool,
        TensorCore, TensorInit, TensorDims, TensorScalar, TensorShape,
        TensorStats, TensorMask, TensorLinAlg, TensorArithmetic, TensorMath,
        TensorComparison, TensorBroadcast,
    };

    #[cfg(feature = "cuda")]
    pub use crate::cuda::CudaBackend;
}
