#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub mod kernels;

// CUDA backend
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

// Prelude
#[cfg(feature = "cuda")]
pub mod prelude {
    pub use crate::cuda::CudaBackend;
    pub use crate::kernels::ElementwiseKernels;
}