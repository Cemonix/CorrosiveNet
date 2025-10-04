/// CUDA backend modules for GPU-accelerated tensor operations
///
/// This module provides:
/// - Error types for CUDA operations
/// - Context management (one context per device, globally cached)
/// - Kernel compilation and caching
/// - High-level backend interface for tensor operations

mod error;
mod context;
mod cache;
mod backend;

pub use backend::CudaBackend;
pub use error::CudaError;

// Internal use only
pub(crate) use context::get_context;
pub(crate) use cache::{get_or_compile_kernel, cache_stats, clear_cache};

