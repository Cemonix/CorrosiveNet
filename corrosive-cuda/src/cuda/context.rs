use cudarc::driver::CudaContext;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::error::CudaError;

/// Global CUDA context manager
///
/// Maintains one context per device to avoid creating multiple contexts.
/// This is a singleton that lives for the entire program lifetime.
struct ContextManagerInner {
    contexts: HashMap<usize, Arc<CudaContext>>,
}

impl ContextManagerInner {
    fn new() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    fn get_or_create(&mut self, device_idx: usize) -> Result<Arc<CudaContext>, CudaError> {
        if let Some(ctx) = self.contexts.get(&device_idx) {
            return Ok(ctx.clone());
        }

        let ctx = CudaContext::new(device_idx)
            .map_err(|e| CudaError::DeviceInit(e.to_string()))?;

        self.contexts.insert(device_idx, ctx.clone());
        Ok(ctx)
    }
}

static CONTEXT_MANAGER: Lazy<Mutex<ContextManagerInner>> = Lazy::new(|| {
    Mutex::new(ContextManagerInner::new())
});

/// Get or create a CUDA context for the specified device
///
/// # Arguments
/// * `device_idx` - CUDA device index (0 for first GPU, 1 for second, etc.)
///
/// # Returns
/// Shared reference to the CUDA context for this device
///
/// # Errors
/// When CUDA device initialization fails or mutex is poisoned
pub fn get_context(device_idx: usize) -> Result<Arc<CudaContext>, CudaError> {
    CONTEXT_MANAGER
        .lock()
        .map_err(|e| CudaError::LockPoisoned(format!("Context manager lock poisoned: {}", e)))?
        .get_or_create(device_idx)
}
