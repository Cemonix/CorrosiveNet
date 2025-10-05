use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use crate::cuda::{CudaBackend, CudaError};

/// Calculate standard 1D launch configuration
///
/// Uses 256 threads per block (common default for simple kernels)
///
/// # Arguments
/// * `n` - Total number of elements to process
///
/// # Returns
/// Launch configuration with appropriate grid and block dimensions
pub fn launch_config_1d(n: usize) -> LaunchConfig {
    let threads_per_block = 256u32;
    let num_blocks = ((n as u32) + threads_per_block - 1) / threads_per_block;

    LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Launch a binary elementwise operation kernel (e.g., add, sub, mul, div)
///
/// This is a generic helper for all binary elementwise operations that follow the pattern:
/// `output[i] = a[i] op b[i]`
///
/// # Arguments
/// * `backend` - CUDA backend for kernel compilation and execution
/// * `kernel_name` - Name of the kernel function in the CUDA source
/// * `kernel_src` - CUDA source code as a string
/// * `a` - First input tensor on GPU
/// * `b` - Second input tensor on GPU
/// * `c` - Output tensor on GPU (will be written to)
/// * `n` - Number of elements
///
/// # Returns
/// Result indicating success or failure
///
/// # Errors
/// When kernel compilation or launch fails
pub fn launch_binary_elementwise<T>(
    backend: &CudaBackend,
    kernel_name: &str,
    kernel_src: &str,
    a: &CudaSlice<T>,
    b: &CudaSlice<T>,
    c: &mut CudaSlice<T>,
    n: usize,
) -> Result<(), CudaError> {
    let module = backend.get_or_compile_kernel(kernel_name, kernel_src)?;
    let cfg = launch_config_1d(n);

    let func = module.load_function(kernel_name)
        .map_err(|e| CudaError::KernelNotFound(format!("{}: {}", kernel_name, e)))?;

    let stream = backend.context().default_stream();

    unsafe {
        stream.launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(c)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| CudaError::KernelLaunch(e.to_string()))?;
    }

    Ok(())
}

/// Launch a unary elementwise operation kernel (e.g., neg, abs, sqrt, exp)
///
/// This is a generic helper for all unary elementwise operations that follow the pattern:
/// `output[i] = op(input[i])`
///
/// # Arguments
/// * `backend` - CUDA backend for kernel compilation and execution
/// * `kernel_name` - Name of the kernel function in the CUDA source
/// * `kernel_src` - CUDA source code as a string
/// * `input` - Input tensor on GPU
/// * `output` - Output tensor on GPU (will be written to)
/// * `n` - Number of elements
///
/// # Returns
/// Result indicating success or failure
///
/// # Errors
/// When kernel compilation or launch fails
pub fn launch_unary_elementwise<T>(
    backend: &CudaBackend,
    kernel_name: &str,
    kernel_src: &str,
    input: &CudaSlice<T>,
    output: &mut CudaSlice<T>,
    n: usize,
) -> Result<(), CudaError> {
    let module = backend.get_or_compile_kernel(kernel_name, kernel_src)?;
    let cfg = launch_config_1d(n);

    let func = module.load_function(kernel_name)
        .map_err(|e| CudaError::KernelNotFound(format!("{}: {}", kernel_name, e)))?;

    let stream = backend.context().default_stream();

    unsafe {
        stream.launch_builder(&func)
            .arg(input)
            .arg(output)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| CudaError::KernelLaunch(e.to_string()))?;
    }

    Ok(())
}

/// Launch a scalar elementwise operation kernel (e.g., tensor + scalar)
///
/// This is a generic helper for all scalar elementwise operations that follow the pattern:
/// `output[i] = input[i] op scalar`
///
/// # Arguments
/// * `backend` - CUDA backend for kernel compilation and execution
/// * `kernel_name` - Name of the kernel function in the CUDA source
/// * `kernel_src` - CUDA source code as a string
/// * `input` - Input tensor on GPU
/// * `scalar` - Scalar value
/// * `output` - Output tensor on GPU (will be written to)
/// * `n` - Number of elements
///
/// # Returns
/// Result indicating success or failure
///
/// # Errors
/// When kernel compilation or launch fails
pub fn launch_scalar_elementwise<T>(
    backend: &CudaBackend,
    kernel_name: &str,
    kernel_src: &str,
    input: &CudaSlice<T>,
    scalar: T,
    output: &mut CudaSlice<T>,
    n: usize,
) -> Result<(), CudaError>
where
    T: cudarc::driver::DeviceRepr,
{
    let module = backend.get_or_compile_kernel(kernel_name, kernel_src)?;
    let cfg = launch_config_1d(n);

    let func = module.load_function(kernel_name)
        .map_err(|e| CudaError::KernelNotFound(format!("{}: {}", kernel_name, e)))?;

    let stream = backend.context().default_stream();

    unsafe {
        stream.launch_builder(&func)
            .arg(input)
            .arg(&scalar)
            .arg(output)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| CudaError::KernelLaunch(e.to_string()))?;
    }

    Ok(())
}

/// Launch a clip operation kernel (takes two scalar parameters: min and max)
///
/// This is a specialized helper for clip operations that follow the pattern:
/// `output[i] = max(min_val, min(input[i], max_val))`
///
/// # Arguments
/// * `backend` - CUDA backend for kernel compilation and execution
/// * `kernel_name` - Name of the kernel function in the CUDA source
/// * `kernel_src` - CUDA source code as a string
/// * `input` - Input tensor on GPU
/// * `min_val` - Minimum clip value
/// * `max_val` - Maximum clip value
/// * `output` - Output tensor on GPU (will be written to)
/// * `n` - Number of elements
///
/// # Returns
/// Result indicating success or failure
///
/// # Errors
/// When kernel compilation or launch fails
pub fn launch_clip_elementwise<T>(
    backend: &CudaBackend,
    kernel_name: &str,
    kernel_src: &str,
    input: &CudaSlice<T>,
    min_val: T,
    max_val: T,
    output: &mut CudaSlice<T>,
    n: usize,
) -> Result<(), CudaError>
where
    T: cudarc::driver::DeviceRepr,
{
    let module = backend.get_or_compile_kernel(kernel_name, kernel_src)?;
    let cfg = launch_config_1d(n);

    let func = module.load_function(kernel_name)
        .map_err(|e| CudaError::KernelNotFound(format!("{}: {}", kernel_name, e)))?;

    let stream = backend.context().default_stream();

    unsafe {
        stream.launch_builder(&func)
            .arg(input)
            .arg(&min_val)
            .arg(&max_val)
            .arg(output)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| CudaError::KernelLaunch(e.to_string()))?;
    }

    Ok(())
}
