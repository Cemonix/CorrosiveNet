use cudarc::driver::{CudaContext, CudaModule};
use cudarc::nvrtc::compile_ptx;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::error::CudaError;

/// Global kernel cache for compiled CUDA modules
///
/// This cache stores compiled CUDA kernels per device to avoid recompilation.
/// Key format: "device_idx:kernel_name"
///
/// Compilation is expensive (~100ms+), so caching provides significant speedup
/// for repeated operations.
struct KernelCacheInner {
    modules: HashMap<String, Arc<CudaModule>>,
}

impl KernelCacheInner {
    fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    fn get(&self, key: &str) -> Option<Arc<CudaModule>> {
        self.modules.get(key).cloned()
    }

    fn insert(&mut self, key: String, module: Arc<CudaModule>) {
        self.modules.insert(key, module);
    }
}

static KERNEL_CACHE: Lazy<Mutex<KernelCacheInner>> = Lazy::new(|| {
    Mutex::new(KernelCacheInner::new())
});

/// Get or compile a CUDA kernel
///
/// This function checks the global kernel cache first. If the kernel is not cached,
/// it compiles the source code and stores the result in the cache for future use.
///
/// # Arguments
/// * `device_idx` - CUDA device index
/// * `kernel_name` - Unique name for this kernel (used as cache key)
/// * `kernel_src` - CUDA source code
/// * `context` - CUDA context to load the module into
///
/// # Returns
/// Cached or newly compiled CUDA module
///
/// # Errors
/// When kernel compilation or loading fails
pub fn get_or_compile_kernel(
    device_idx: usize,
    kernel_name: &str,
    kernel_src: &str,
    context: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, CudaError> {
    let cache_key = format!("{}:{}", device_idx, kernel_name);

    // Check cache first
    {
        let cache = KERNEL_CACHE
            .lock()
            .map_err(|e| CudaError::LockPoisoned(format!("Kernel cache lock poisoned: {}", e)))?;
        if let Some(module) = cache.get(&cache_key) {
            return Ok(module);
        }
    }

    // Not in cache - compile it
    let ptx = compile_ptx(kernel_src)
        .map_err(|e| CudaError::KernelCompilation(e.to_string()))?;

    let module = context.load_module(ptx)
        .map_err(|e| CudaError::KernelLoad(e.to_string()))?;

    // Store in cache
    let mut cache = KERNEL_CACHE
        .lock()
        .map_err(|e| CudaError::LockPoisoned(format!("Kernel cache lock poisoned: {}", e)))?;
    cache.insert(cache_key, module.clone());

    Ok(module)
}

/// Get cache statistics (for debugging/monitoring)
///
/// # Errors
/// When mutex is poisoned
#[allow(dead_code)]
pub fn cache_stats() -> Result<usize, CudaError> {
    let cache = KERNEL_CACHE
        .lock()
        .map_err(|e| CudaError::LockPoisoned(format!("Kernel cache lock poisoned: {}", e)))?;
    Ok(cache.modules.len())
}

/// Clear the kernel cache (useful for testing)
///
/// # Errors
/// When mutex is poisoned
#[allow(dead_code)]
pub fn clear_cache() -> Result<(), CudaError> {
    let mut cache = KERNEL_CACHE
        .lock()
        .map_err(|e| CudaError::LockPoisoned(format!("Kernel cache lock poisoned: {}", e)))?;
    cache.modules.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_caching() {
        let backend = match crate::cuda::CudaBackend::new(0) {
            Ok(b) => b,
            Err(_) => {
                println!("CUDA not available, skipping test");
                return;
            }
        };

        let kernel_src = include_str!("../../kernels/add.cu");

        // Clear cache
        clear_cache().unwrap();
        assert_eq!(cache_stats().unwrap(), 0);

        // First compilation
        let _ = backend.get_or_compile_kernel("test_kernel", kernel_src).unwrap();
        assert_eq!(cache_stats().unwrap(), 1);

        // Second call should use cache
        let _ = backend.get_or_compile_kernel("test_kernel", kernel_src).unwrap();
        assert_eq!(cache_stats().unwrap(), 1); // Still 1, not 2

        // Different kernel name
        let _ = backend.get_or_compile_kernel("test_kernel_2", kernel_src).unwrap();
        assert_eq!(cache_stats().unwrap(), 2);
    }
}
