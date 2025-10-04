use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, CudaModule, PushKernelArg};
use std::sync::Arc;

use super::cache::get_or_compile_kernel;
use super::context::get_context;
use super::error::CudaError;

/// CUDA backend for tensor operations
///
/// Provides high-level interface for CUDA operations including:
/// - Context management
/// - Kernel compilation and caching
/// - Kernel execution
pub struct CudaBackend {
    context: Arc<CudaContext>,
    device_idx: usize,
}

impl CudaBackend {
    /// Initialize CUDA backend with specified device index
    ///
    /// Uses global context manager to reuse contexts across tensor operations.
    ///
    /// # Arguments
    /// * `device_idx` - CUDA device index (typically 0 for single GPU systems)
    ///
    /// # Returns
    /// Initialized CUDA backend
    ///
    /// # Errors
    /// When CUDA device initialization fails
    pub fn new(device_idx: usize) -> Result<Self, CudaError> {
        let context = get_context(device_idx)?;

        Ok(Self {
            context,
            device_idx,
        })
    }

    /// Get reference to the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get device index
    pub fn device_idx(&self) -> usize {
        self.device_idx
    }

    /// Get or compile a CUDA kernel from source
    ///
    /// This method checks the global kernel cache first. If the kernel is not cached,
    /// it compiles it and stores it in the cache for future use.
    ///
    /// # Arguments
    /// * `kernel_name` - Unique name for this kernel (used as cache key)
    /// * `kernel_src` - CUDA source code
    ///
    /// # Returns
    /// Cached or newly compiled CUDA module
    ///
    /// # Errors
    /// When kernel compilation or loading fails
    pub fn get_or_compile_kernel(
        &self,
        kernel_name: &str,
        kernel_src: &str,
    ) -> Result<Arc<CudaModule>, CudaError> {
        get_or_compile_kernel(self.device_idx, kernel_name, kernel_src, &self.context)
    }

    /// Load kernel from a file path
    ///
    /// # Arguments
    /// * `kernel_path` - Path to the .cu file
    /// * `kernel_name` - Name for caching
    ///
    /// # Returns
    /// The loaded CUDA module
    ///
    /// # Errors
    /// When file reading, kernel compilation, or loading fails
    pub fn load_kernel_from_file(
        &self,
        kernel_path: &str,
        kernel_name: &str,
    ) -> Result<Arc<CudaModule>, CudaError> {
        let kernel_src = std::fs::read_to_string(kernel_path)
            .map_err(|e| CudaError::FileRead(e.to_string()))?;
        self.get_or_compile_kernel(kernel_name, &kernel_src)
    }

    /// Launch element-wise addition kernel for f32 tensors
    ///
    /// # Arguments
    /// * `module` - The loaded CUDA module containing the kernel
    /// * `a` - First input array on GPU
    /// * `b` - Second input array on GPU
    /// * `c` - Output array on GPU
    /// * `n` - Number of elements
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Errors
    /// When kernel launch fails
    pub fn elementwise_add_f32(
        &self,
        module: &Arc<CudaModule>,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        let threads_per_block = 256u32;
        let num_blocks = ((n as u32) + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = module.load_function("elementwise_add_f32")
            .map_err(|e| CudaError::KernelNotFound(format!("elementwise_add_f32: {}", e)))?;

        let stream = self.context.default_stream();
        let mut builder = stream.launch_builder(&func);
        builder.arg(a);
        builder.arg(b);
        builder.arg(c);
        builder.arg(&n);

        unsafe {
            builder.launch(cfg)
                .map_err(|e| CudaError::KernelLaunch(e.to_string()))?;
        }

        Ok(())
    }
}
