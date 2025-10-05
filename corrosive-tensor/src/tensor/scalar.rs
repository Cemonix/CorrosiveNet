use super::{Tensor, TensorNum, TensorStorage, TensorCore, Device};

#[cfg(feature = "cuda")]
use super::TensorError;

pub trait TensorScalar<T> {
    fn scalar_add(&self, scalar: T) -> Tensor<T>;
    fn scalar_add_mut(&mut self, scalar: T);
    fn scalar_sub(&self, scalar: T) -> Tensor<T>;
    fn scalar_sub_mut(&mut self, scalar: T);
    fn scalar_mul(&self, scalar: T) -> Tensor<T>;
    fn scalar_mul_mut(&mut self, scalar: T);
    fn scalar_div(&self, scalar: T) -> Tensor<T>;
    fn scalar_div_mut(&mut self, scalar: T);
}

// Trait implementation for f32
impl TensorScalar<f32> for Tensor<f32> {
    fn scalar_add(&self, scalar: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_scalar_add(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_add(scalar).expect("CUDA scalar_add failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_add_mut(&mut self, scalar: f32) {
        match self.device() {
            Device::CPU => self.cpu_scalar_add_mut(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_add_mut(scalar).expect("CUDA scalar_add_mut failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_sub(&self, scalar: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_scalar_sub(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_sub(scalar).expect("CUDA scalar_sub failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_sub_mut(&mut self, scalar: f32) {
        match self.device() {
            Device::CPU => self.cpu_scalar_sub_mut(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_sub_mut(scalar).expect("CUDA scalar_sub_mut failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_mul(&self, scalar: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_scalar_mul(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_mul(scalar).expect("CUDA scalar_mul failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_mul_mut(&mut self, scalar: f32) {
        match self.device() {
            Device::CPU => self.cpu_scalar_mul_mut(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_mul_mut(scalar).expect("CUDA scalar_mul_mut failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_div(&self, scalar: f32) -> Tensor<f32> {
        match self.device() {
            Device::CPU => self.cpu_scalar_div(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_div(scalar).expect("CUDA scalar_div failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_div_mut(&mut self, scalar: f32) {
        match self.device() {
            Device::CPU => self.cpu_scalar_div_mut(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_div_mut(scalar).expect("CUDA scalar_div_mut failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }
}

// Trait implementation for f64
impl TensorScalar<f64> for Tensor<f64> {
    fn scalar_add(&self, scalar: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_scalar_add(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_add(scalar).expect("CUDA scalar_add failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_add_mut(&mut self, scalar: f64) {
        match self.device() {
            Device::CPU => self.cpu_scalar_add_mut(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_add_mut(scalar).expect("CUDA scalar_add_mut failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_sub(&self, scalar: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_scalar_sub(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_sub(scalar).expect("CUDA scalar_sub failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_sub_mut(&mut self, scalar: f64) {
        match self.device() {
            Device::CPU => self.cpu_scalar_sub_mut(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_sub_mut(scalar).expect("CUDA scalar_sub_mut failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_mul(&self, scalar: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_scalar_mul(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_mul(scalar).expect("CUDA scalar_mul failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_mul_mut(&mut self, scalar: f64) {
        match self.device() {
            Device::CPU => self.cpu_scalar_mul_mut(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_mul_mut(scalar).expect("CUDA scalar_mul_mut failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_div(&self, scalar: f64) -> Tensor<f64> {
        match self.device() {
            Device::CPU => self.cpu_scalar_div(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_div(scalar).expect("CUDA scalar_div failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }

    fn scalar_div_mut(&mut self, scalar: f64) {
        match self.device() {
            Device::CPU => self.cpu_scalar_div_mut(scalar),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_scalar_div_mut(scalar).expect("CUDA scalar_div_mut failed"),
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => panic!("CUDA support not compiled. Rebuild with --features cuda"),
        }
    }
}

// CPU implementations
impl<T: TensorNum> Tensor<T> {
    fn cpu_scalar_add(&self, scalar: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| x + scalar).collect();
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

    fn cpu_scalar_add_mut(&mut self, scalar: T) {
        match &mut self.storage {
            TensorStorage::CPU(data) => {
                for x in data.iter_mut() {
                    *x = *x + scalar;
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_scalar_sub(&self, scalar: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| x - scalar).collect();
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

    fn cpu_scalar_sub_mut(&mut self, scalar: T) {
        match &mut self.storage {
            TensorStorage::CPU(data) => {
                for x in data.iter_mut() {
                    *x = *x - scalar;
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_scalar_mul(&self, scalar: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| x * scalar).collect();
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

    fn cpu_scalar_mul_mut(&mut self, scalar: T) {
        match &mut self.storage {
            TensorStorage::CPU(data) => {
                for x in data.iter_mut() {
                    *x = *x * scalar;
                }
            }
            #[cfg(feature = "cuda")]
            _ => panic!("Expected CPU tensor"),
        }
    }

    fn cpu_scalar_div(&self, scalar: T) -> Tensor<T> {
        match &self.storage {
            TensorStorage::CPU(data) => {
                let result: Vec<T> = data.iter().map(|&x| x / scalar).collect();
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

    fn cpu_scalar_div_mut(&mut self, scalar: T) {
        match &mut self.storage {
            TensorStorage::CPU(data) => {
                for x in data.iter_mut() {
                    *x = *x / scalar;
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
    fn cuda_scalar_add(&self, scalar: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::add_scalar_f32(&backend, buffer, scalar, &mut output, n)
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

    fn cuda_scalar_add_mut(&mut self, scalar: f32) -> Result<(), TensorError> {
        let result = self.cuda_scalar_add(scalar)?;
        *self = result;
        Ok(())
    }

    fn cuda_scalar_sub(&self, scalar: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::sub_scalar_f32(&backend, buffer, scalar, &mut output, n)
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

    fn cuda_scalar_sub_mut(&mut self, scalar: f32) -> Result<(), TensorError> {
        let result = self.cuda_scalar_sub(scalar)?;
        *self = result;
        Ok(())
    }

    fn cuda_scalar_mul(&self, scalar: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::mul_scalar_f32(&backend, buffer, scalar, &mut output, n)
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

    fn cuda_scalar_mul_mut(&mut self, scalar: f32) -> Result<(), TensorError> {
        let result = self.cuda_scalar_mul(scalar)?;
        *self = result;
        Ok(())
    }

    fn cuda_scalar_div(&self, scalar: f32) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::div_scalar_f32(&backend, buffer, scalar, &mut output, n)
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

    fn cuda_scalar_div_mut(&mut self, scalar: f32) -> Result<(), TensorError> {
        let result = self.cuda_scalar_div(scalar)?;
        *self = result;
        Ok(())
    }
}

// CUDA implementations for f64
#[cfg(feature = "cuda")]
impl Tensor<f64> {
    fn cuda_scalar_add(&self, scalar: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::add_scalar_f64(&backend, buffer, scalar, &mut output, n)
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

    fn cuda_scalar_add_mut(&mut self, scalar: f64) -> Result<(), TensorError> {
        let result = self.cuda_scalar_add(scalar)?;
        *self = result;
        Ok(())
    }

    fn cuda_scalar_sub(&self, scalar: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::sub_scalar_f64(&backend, buffer, scalar, &mut output, n)
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

    fn cuda_scalar_sub_mut(&mut self, scalar: f64) -> Result<(), TensorError> {
        let result = self.cuda_scalar_sub(scalar)?;
        *self = result;
        Ok(())
    }

    fn cuda_scalar_mul(&self, scalar: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::mul_scalar_f64(&backend, buffer, scalar, &mut output, n)
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

    fn cuda_scalar_mul_mut(&mut self, scalar: f64) -> Result<(), TensorError> {
        let result = self.cuda_scalar_mul(scalar)?;
        *self = result;
        Ok(())
    }

    fn cuda_scalar_div(&self, scalar: f64) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match &self.storage {
            TensorStorage::CUDA { context, buffer, device_idx } => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut output = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::div_scalar_f64(&backend, buffer, scalar, &mut output, n)
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

    fn cuda_scalar_div_mut(&mut self, scalar: f64) -> Result<(), TensorError> {
        let result = self.cuda_scalar_div(scalar)?;
        *self = result;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Device, Tensor, TensorCore, TensorScalar, TensorInit};

    #[test]
    fn test_scalar_operations() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();

        // Scalar addition
        let result = tensor.scalar_add(10.0);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 14.0);

        // Scalar multiplication
        let result = tensor.scalar_mul(2.0);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 8.0);

        // Scalar division
        let result = tensor.scalar_div(2.0);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 0.5);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 2.0);
    }

    #[test]
    fn test_scalar_operations_mut() {
        let mut tensor = Tensor::<f32>::from_data(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2], Device::CPU).unwrap();

        tensor.scalar_mul_mut(0.5);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 4.0);

        tensor.scalar_add_mut(1.0);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 5.0);
    }
}
