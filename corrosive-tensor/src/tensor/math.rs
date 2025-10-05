use super::{Tensor, TensorNum, TensorFloat, TensorStorage, TensorCore, Device};

#[cfg(feature = "cuda")]
use super::TensorError;

pub trait TensorMath<T> {
    fn exp(&self) -> Tensor<T> where T: TensorFloat;
    fn log(&self) -> Tensor<T> where T: TensorFloat;
    fn sqrt(&self) -> Tensor<T> where T: TensorFloat;
    fn square(&self) -> Tensor<T>;
    fn abs(&self) -> Tensor<T> where T: TensorFloat;
    fn pow(&self, exponent: T) -> Tensor<T> where T: TensorFloat;
}

// Trait implementation for f32
impl TensorMath<f32> for Tensor<f32> {
    fn exp(&self) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_exp(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_exp().expect("CUDA exp failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn log(&self) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_log(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_log().expect("CUDA log failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn sqrt(&self) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_sqrt(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_sqrt().expect("CUDA sqrt failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn square(&self) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_square(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_square().expect("CUDA square failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn abs(&self) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_abs(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_abs().expect("CUDA abs failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn pow(&self, exponent: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_pow(exponent),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_pow(exponent).expect("CUDA pow failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }
}

// Trait implementation for f64
impl TensorMath<f64> for Tensor<f64> {
    fn exp(&self) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_exp(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_exp().expect("CUDA exp failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn log(&self) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_log(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_log().expect("CUDA log failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn sqrt(&self) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_sqrt(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_sqrt().expect("CUDA sqrt failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn square(&self) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_square(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_square().expect("CUDA square failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn abs(&self) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_abs(),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_abs().expect("CUDA abs failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn pow(&self, exponent: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_pow(exponent),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_pow(exponent).expect("CUDA pow failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }
}

// CPU implementations
impl<T: TensorNum> Tensor<T> {
    fn cpu_exp(&self) -> Tensor<T>
    where
        T: TensorFloat,
    {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| x.exp()).collect();
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

    fn cpu_log(&self) -> Tensor<T>
    where
        T: TensorFloat,
    {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| x.ln()).collect();
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

    fn cpu_sqrt(&self) -> Tensor<T>
    where
        T: TensorFloat,
    {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| x.sqrt()).collect();
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

    fn cpu_square(&self) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| x * x).collect();
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

    fn cpu_abs(&self) -> Tensor<T>
    where
        T: TensorFloat,
    {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| x.abs()).collect();
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

    fn cpu_pow(&self, exponent: T) -> Tensor<T>
    where
        T: TensorFloat,
    {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| x.powf(exponent)).collect();
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
    fn cuda_exp(&self) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::exp_f32(&backend, buffer, &mut output, n)
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

    fn cuda_log(&self) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::log_f32(&backend, buffer, &mut output, n)
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

    fn cuda_sqrt(&self) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::sqrt_f32(&backend, buffer, &mut output, n)
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

    fn cuda_square(&self) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::square_f32(&backend, buffer, &mut output, n)
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

    fn cuda_abs(&self) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::abs_f32(&backend, buffer, &mut output, n)
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

    fn cuda_pow(&self, exponent: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::pow_scalar_f32(&backend, buffer, exponent, &mut output, n)
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
    fn cuda_exp(&self) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::exp_f64(&backend, buffer, &mut output, n)
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

    fn cuda_log(&self) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::log_f64(&backend, buffer, &mut output, n)
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

    fn cuda_sqrt(&self) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::sqrt_f64(&backend, buffer, &mut output, n)
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

    fn cuda_square(&self) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::square_f64(&backend, buffer, &mut output, n)
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

    fn cuda_abs(&self) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::abs_f64(&backend, buffer, &mut output, n)
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

    fn cuda_pow(&self, exponent: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::pow_scalar_f64(&backend, buffer, exponent, &mut output, n)
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
    use crate::tensor::{Device, Tensor, TensorCore, TensorMath, TensorInit};


    #[test]
    fn test_square() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.square();

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0);  // 1^2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);  // 2^2
        assert_eq!(*result.get(&[1, 0]).unwrap(), 9.0);  // 3^2
        assert_eq!(*result.get(&[1, 1]).unwrap(), 16.0); // 4^2
    }

    #[test]
    fn test_sqrt() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.sqrt();

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0); // sqrt(1)
        assert_eq!(*result.get(&[0, 1]).unwrap(), 2.0); // sqrt(4)
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0); // sqrt(9)
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0); // sqrt(16)
    }

    #[test]
    fn test_exp() {
        let tensor = Tensor::<f32>::from_data(vec![0.0, 1.0], vec![1, 2], Device::CPU).unwrap();
        let result = tensor.exp();

        assert!((result.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-6); // exp(0) ≈ 1
        assert!((result.get(&[0, 1]).unwrap() - 2.71828).abs() < 1e-4); // exp(1) ≈ e
    }

    #[test]
    fn test_log() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, std::f32::consts::E], vec![1, 2], Device::CPU).unwrap();
        let result = tensor.log();

        assert!((result.get(&[0, 0]).unwrap() - 0.0).abs() < 1e-6); // ln(1) = 0
        assert!((result.get(&[0, 1]).unwrap() - 1.0).abs() < 1e-6); // ln(e) = 1
    }

    #[test]
    fn test_abs() {
        let tensor = Tensor::<f32>::from_data(vec![-2.0, -1.0, 0.0, 1.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.abs();

        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0); // |-2|
        assert_eq!(*result.get(&[0, 1]).unwrap(), 1.0); // |-1|
        assert_eq!(*result.get(&[1, 0]).unwrap(), 0.0); // |0|
        assert_eq!(*result.get(&[1, 1]).unwrap(), 1.0); // |1|
    }

    #[test]
    fn test_pow() {
        let tensor = Tensor::<f32>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.pow(3.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 8.0);   // 2^3
        assert_eq!(*result.get(&[0, 1]).unwrap(), 27.0);  // 3^3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 64.0);  // 4^3
        assert_eq!(*result.get(&[1, 1]).unwrap(), 125.0); // 5^3
    }
}