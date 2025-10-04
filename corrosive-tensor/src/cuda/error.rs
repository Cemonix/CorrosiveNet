use std::error::Error;
use std::fmt;

/// CUDA operation errors
#[derive(Debug)]
pub enum CudaError {
    DeviceInit(String),
    KernelCompilation(String),
    KernelLoad(String),
    KernelLaunch(String),
    KernelNotFound(String),
    FileRead(String),
    LockPoisoned(String),
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
            CudaError::LockPoisoned(msg) => write!(f, "Mutex lock poisoned: {}", msg),
        }
    }
}

impl Error for CudaError {}
