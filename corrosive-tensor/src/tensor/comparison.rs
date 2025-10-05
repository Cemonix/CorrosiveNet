use super::{Tensor, TensorNum, TensorStorage, TensorCore, Device};

#[cfg(feature = "cuda")]
use super::TensorError;

pub trait TensorComparison<T> {
    fn greater_than(&self, threshold: T) -> Tensor<T>;
    fn greater_equal(&self, threshold: T) -> Tensor<T>;
    fn less_than(&self, threshold: T) -> Tensor<T>;
    fn less_equal(&self, threshold: T) -> Tensor<T>;
    fn equal(&self, threshold: T) -> Tensor<T>;
    fn not_equal(&self, threshold: T) -> Tensor<T>;
    fn clip_max(&self, threshold: T) -> Tensor<T>;
    fn clip_min(&self, threshold: T) -> Tensor<T>;
    fn clip(&self, min_val: T, max_val: T) -> Tensor<T>;
}

// Trait implementation for f32
impl TensorComparison<f32> for Tensor<f32> {
    fn greater_than(&self, threshold: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_greater_than(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_greater_than(threshold).expect("CUDA greater_than failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn greater_equal(&self, threshold: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_greater_equal(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_greater_equal(threshold).expect("CUDA greater_equal failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn less_than(&self, threshold: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_less_than(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_less_than(threshold).expect("CUDA less_than failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn less_equal(&self, threshold: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_less_equal(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_less_equal(threshold).expect("CUDA less_equal failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn equal(&self, threshold: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_equal(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_equal(threshold).expect("CUDA equal failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn not_equal(&self, threshold: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_not_equal(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_not_equal(threshold).expect("CUDA not_equal failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn clip_max(&self, threshold: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_clip_max(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_clip_max(threshold).expect("CUDA clip_max failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn clip_min(&self, threshold: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_clip_min(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_clip_min(threshold).expect("CUDA clip_min failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn clip(&self, min_val: f32, max_val: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_clip(min_val, max_val),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_clip(min_val, max_val).expect("CUDA clip failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }
}

// Trait implementation for f64
impl TensorComparison<f64> for Tensor<f64> {
    fn greater_than(&self, threshold: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_greater_than(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_greater_than(threshold).expect("CUDA greater_than failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn greater_equal(&self, threshold: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_greater_equal(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_greater_equal(threshold).expect("CUDA greater_equal failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn less_than(&self, threshold: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_less_than(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_less_than(threshold).expect("CUDA less_than failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn less_equal(&self, threshold: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_less_equal(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_less_equal(threshold).expect("CUDA less_equal failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn equal(&self, threshold: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_equal(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_equal(threshold).expect("CUDA equal failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn not_equal(&self, threshold: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_not_equal(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_not_equal(threshold).expect("CUDA not_equal failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn clip_max(&self, threshold: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_clip_max(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_clip_max(threshold).expect("CUDA clip_max failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn clip_min(&self, threshold: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_clip_min(threshold),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_clip_min(threshold).expect("CUDA clip_min failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn clip(&self, min_val: f64, max_val: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_clip(min_val, max_val),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_clip(min_val, max_val).expect("CUDA clip failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }
}

// CPU implementations
impl<T: TensorNum> Tensor<T> {
    fn cpu_greater_than(&self, threshold: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let zero = T::zero();
                let one = T::one();
                let result: Vec<T> = data.iter().map(|&x| if x > threshold { one } else { zero }).collect();
                Tensor {
                    storage: TensorStorage::CPU(result),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_greater_equal(&self, threshold: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let zero = T::zero();
                let one = T::one();
                let result: Vec<T> = data.iter().map(|&x| if x >= threshold { one } else { zero }).collect();
                Tensor {
                    storage: TensorStorage::CPU(result),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_less_than(&self, threshold: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let zero = T::zero();
                let one = T::one();
                let result: Vec<T> = data.iter().map(|&x| if x < threshold { one } else { zero }).collect();
                Tensor {
                    storage: TensorStorage::CPU(result),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_less_equal(&self, threshold: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let zero = T::zero();
                let one = T::one();
                let result: Vec<T> = data.iter().map(|&x| if x <= threshold { one } else { zero }).collect();
                Tensor {
                    storage: TensorStorage::CPU(result),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_equal(&self, threshold: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let zero = T::zero();
                let one = T::one();
                let result: Vec<T> = data.iter().map(|&x| if x == threshold { one } else { zero }).collect();
                Tensor {
                    storage: TensorStorage::CPU(result),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_not_equal(&self, threshold: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let zero = T::zero();
                let one = T::one();
                let result: Vec<T> = data.iter().map(|&x| if x != threshold { one } else { zero }).collect();
                Tensor {
                    storage: TensorStorage::CPU(result),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_clip_max(&self, threshold: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| if x > threshold { threshold } else { x }).collect();
                Tensor {
                    storage: TensorStorage::CPU(result),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_clip_min(&self, threshold: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| if x < threshold { threshold } else { x }).collect();
                Tensor {
                    storage: TensorStorage::CPU(result),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_clip(&self, min_val: T, max_val: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter()
                    .map(|&x| {
                        if x < min_val {
                            min_val
                        } else if x > max_val {
                            max_val
                        } else {
                            x
                        }
                    })
                    .collect();
                Tensor {
                    storage: TensorStorage::CPU(result),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }
}

// CUDA implementations for f32
#[cfg(feature = "cuda")]
impl Tensor<f32> {
    fn cuda_greater_than(&self, threshold: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::greater_than_f32(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_greater_equal(&self, threshold: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::greater_equal_f32(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_less_than(&self, threshold: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::less_than_f32(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_less_equal(&self, threshold: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::less_equal_f32(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_equal(&self, threshold: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::equal_f32(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_not_equal(&self, threshold: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::not_equal_f32(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_clip_max(&self, threshold: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::clip_max_f32(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_clip_min(&self, threshold: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::clip_min_f32(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_clip(&self, min_val: f32, max_val: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::clip_f32(&backend, buffer, min_val, max_val, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }
}

// CUDA implementations for f64
#[cfg(feature = "cuda")]
impl Tensor<f64> {
    fn cuda_greater_than(&self, threshold: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::greater_than_f64(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_greater_equal(&self, threshold: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::greater_equal_f64(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_less_than(&self, threshold: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::less_than_f64(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_less_equal(&self, threshold: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::less_equal_f64(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_equal(&self, threshold: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::equal_f64(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_not_equal(&self, threshold: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::not_equal_f64(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_clip_max(&self, threshold: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::clip_max_f64(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_clip_min(&self, threshold: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::clip_min_f64(&backend, buffer, threshold, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }

    fn cuda_clip(&self, min_val: f64, max_val: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::clip_f64(&backend, buffer, min_val, max_val, &mut output, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: output,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensor")),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Device, Tensor, TensorComparison, TensorCore, TensorInit};


    #[test]
    fn test_greater_than() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 3.0, 5.0, 2.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.greater_than(3.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 0.0); // 1 > 3? No
        assert_eq!(*result.get(&[0, 1]).unwrap(), 0.0); // 3 > 3? No
        assert_eq!(*result.get(&[1, 0]).unwrap(), 1.0); // 5 > 3? Yes
        assert_eq!(*result.get(&[1, 1]).unwrap(), 0.0); // 2 > 3? No
    }

    #[test]
    fn test_greater_equal() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 3.0, 5.0, 2.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.greater_equal(3.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 0.0); // 1 >= 3? No
        assert_eq!(*result.get(&[0, 1]).unwrap(), 1.0); // 3 >= 3? Yes
        assert_eq!(*result.get(&[1, 0]).unwrap(), 1.0); // 5 >= 3? Yes
        assert_eq!(*result.get(&[1, 1]).unwrap(), 0.0); // 2 >= 3? No
    }

    #[test]
    fn test_less_than() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 3.0, 5.0, 2.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.less_than(3.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0); // 1 < 3? Yes
        assert_eq!(*result.get(&[0, 1]).unwrap(), 0.0); // 3 < 3? No
        assert_eq!(*result.get(&[1, 0]).unwrap(), 0.0); // 5 < 3? No
        assert_eq!(*result.get(&[1, 1]).unwrap(), 1.0); // 2 < 3? Yes
    }

    #[test]
    fn test_equal() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 3.0, 3.0, 2.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.equal(3.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 0.0); // 1 == 3? No
        assert_eq!(*result.get(&[0, 1]).unwrap(), 1.0); // 3 == 3? Yes
        assert_eq!(*result.get(&[1, 0]).unwrap(), 1.0); // 3 == 3? Yes
        assert_eq!(*result.get(&[1, 1]).unwrap(), 0.0); // 2 == 3? No
    }

    #[test]
    fn test_clip_max() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 5.0, 3.0, 8.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.clip_max(4.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0); // 1 < 4, unchanged
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0); // 5 > 4, clipped to 4
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0); // 3 < 4, unchanged
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0); // 8 > 4, clipped to 4
    }

    #[test]
    fn test_clip_min() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 5.0, 3.0, 8.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.clip_min(4.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 4.0); // 1 < 4, clipped to 4
        assert_eq!(*result.get(&[0, 1]).unwrap(), 5.0); // 5 > 4, unchanged
        assert_eq!(*result.get(&[1, 0]).unwrap(), 4.0); // 3 < 4, clipped to 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 8.0); // 8 > 4, unchanged
    }

    #[test]
    fn test_clip() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 5.0, 3.0, 8.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.clip(2.0, 6.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0); // 1 < 2, clipped to 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 5.0); // 5 in [2,6], unchanged
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0); // 3 in [2,6], unchanged
        assert_eq!(*result.get(&[1, 1]).unwrap(), 6.0); // 8 > 6, clipped to 6
    }
}
