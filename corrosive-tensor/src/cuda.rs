use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, CudaModule, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::error::Error;
use std::sync::Arc;
use std::fmt;

#[derive(Debug)]
pub enum CudaError {
    DeviceInit(String),
    KernelCompilation(String),
    KernelLoad(String),
    KernelLaunch(String),
    KernelNotFound(String),
    FileRead(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CudaError::DeviceInit(msg) => write!(f, "CUDA device initialization failed: {}", msg),
            CudaError::KernelCompilation(msg) => write!(f, "Kernel compilation failed: {}", msg),
            CudaError::KernelLoad(msg) => write!(f, "Kernel load failed: {}", msg),
            CudaError::KernelLaunch(msg) => write!(f, "Kernel launch failed: {}", msg),
            CudaError::KernelNotFound(msg) => write!(f, "Kernel not found: {}", msg),
            CudaError::FileRead(msg) => write!(f, "File read failed: {}", msg),
        }
    }
}

impl Error for CudaError {}

/// CUDA backend for tensor operations
pub struct CudaBackend {
    context: Arc<CudaContext>,
}

impl CudaBackend {
    /// Initialize CUDA backend with specified device index
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
        let context = CudaContext::new(device_idx)
            .map_err(|e| CudaError::DeviceInit(e.to_string()))?;
        Ok(Self { context })
    }

    /// Get reference to the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Compile and load a CUDA kernel from source
    ///
    /// # Arguments
    /// * `kernel_src` - CUDA kernel source code as string
    /// * `module_name` - Name to give the compiled module
    /// * `kernel_names` - Names of kernel functions to load from the module
    ///
    /// # Returns
    /// The loaded CUDA module
    ///
    /// # Errors
    /// When kernel compilation or loading fails
    pub fn load_kernel(
        &self,
        kernel_src: &str,
        _module_name: &str,
        _kernel_names: &[&str],
    ) -> Result<Arc<CudaModule>, CudaError> {
        let ptx = compile_ptx(kernel_src)
            .map_err(|e| CudaError::KernelCompilation(e.to_string()))?;

        let module = self.context.load_module(ptx)
            .map_err(|e| CudaError::KernelLoad(e.to_string()))?;

        Ok(module)
    }

    /// Load kernel from a file path
    ///
    /// # Arguments
    /// * `kernel_path` - Path to the .cu file
    /// * `module_name` - Name to give the compiled module
    /// * `kernel_names` - Names of kernel functions to load from the module
    ///
    /// # Returns
    /// The loaded CUDA module
    ///
    /// # Errors
    /// When file reading, kernel compilation, or loading fails
    pub fn load_kernel_from_file(
        &self,
        kernel_path: &str,
        module_name: &str,
        kernel_names: &[&str],
    ) -> Result<Arc<CudaModule>, CudaError> {
        let kernel_src = std::fs::read_to_string(kernel_path)
            .map_err(|e| CudaError::FileRead(e.to_string()))?;
        self.load_kernel(&kernel_src, module_name, kernel_names)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_backend_init() {
        match CudaBackend::new(0) {
            Ok(backend) => {
                let kernel_src = include_str!("../kernels/add.cu");
                assert!(backend.load_kernel(kernel_src, "add_module", &["elementwise_add_f32"]).is_ok());
            }
            Err(e) => {
                println!("CUDA not available: {}", e);
            }
        }
    }

    #[test]
    fn test_elementwise_add() {
        let backend = match CudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let kernel_src = include_str!("../kernels/add.cu");
        let module = backend.load_kernel(kernel_src, "add_module", &["elementwise_add_f32"]).unwrap();

        let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_host = vec![5.0f32, 6.0, 7.0, 8.0];
        let n = a_host.len();

        let stream = backend.context().default_stream();
        let a_dev = stream.memcpy_stod(&a_host).unwrap();
        let b_dev = stream.memcpy_stod(&b_host).unwrap();
        let mut c_dev = stream.alloc_zeros::<f32>(n).unwrap();

        backend.elementwise_add_f32(&module, &a_dev, &b_dev, &mut c_dev, n).unwrap();

        let c_host = stream.memcpy_dtov(&c_dev).unwrap();

        assert_eq!(c_host[0], 6.0);
        assert_eq!(c_host[1], 8.0);
        assert_eq!(c_host[2], 10.0);
        assert_eq!(c_host[3], 12.0);
    }
}
