use super::{Tensor, TensorError, TensorNum, TensorStorage, TensorCore, Device};

pub trait TensorArithmetic<T> {
    fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn add_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError>;
    fn sub(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn sub_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError>;
    fn elementwise_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn elementwise_mul_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError>;
    fn elementwise_div(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn elementwise_div_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError>;
}

impl<T> TensorArithmetic<T> for Tensor<T>
where
    T: TensorNum,
{
    /// Element-wise addition of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to add to this one
    ///
    /// # Returns
    /// A new tensor containing the element-wise sum
    ///
    /// # Errors
    /// When shapes do not match or tensors are on different devices
    fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for addition"));
        }

        if !self.has_same_device(other) {
            return Err(TensorError::new(
                "Tensors must be on the same device. Use .to(device) to move tensors."
            ));
        }

        match self.device() {
            Device::CPU => self.cpu_add(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_add(other),
            #[cfg(not(feature = "cuda"))] Device::CUDA(_) => {
                Err(TensorError::new("CUDA support not compiled. Rebuild with --features cuda"))
            }
        }
    }

    /// In-place element-wise addition of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to add to this one
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match or tensors are on different devices
    fn add_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for addition"));
        }

        if !self.has_same_device(other) {
            return Err(TensorError::new(
                "Tensors must be on the same device. Use .to(device) to move tensors."
            ));
        }

        match self.device() {
            Device::CPU => self.cpu_add_mut(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_add_mut(other),
            #[cfg(not(feature = "cuda"))] Device::CUDA(_) => {
                Err(TensorError::new("CUDA support not compiled. Rebuild with --features cuda"))
            }
        }
    }

    /// Element-wise subtraction of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from this one
    ///
    /// # Returns
    /// A new tensor containing the element-wise difference
    ///
    /// # Errors
    /// When shapes do not match or tensors are on different devices
    fn sub(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for subtraction"));
        }

        if !self.has_same_device(other) {
            return Err(TensorError::new(
                "Tensors must be on the same device. Use .to(device) to move tensors."
            ));
        }

        match self.device() {
            Device::CPU => self.cpu_sub(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_sub(other),
            #[cfg(not(feature = "cuda"))] Device::CUDA(_) => {
                Err(TensorError::new("CUDA support not compiled. Rebuild with --features cuda"))
            }
        }
    }

    /// In-place element-wise subtraction of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from this one
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match or tensors are on different devices
    fn sub_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for subtraction"));
        }

        if !self.has_same_device(other) {
            return Err(TensorError::new(
                "Tensors must be on the same device. Use .to(device) to move tensors."
            ));
        }

        match self.device() {
            Device::CPU => self.cpu_sub_mut(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_sub_mut(other),
            #[cfg(not(feature = "cuda"))] Device::CUDA(_) => {
                Err(TensorError::new("CUDA support not compiled. Rebuild with --features cuda"))
            }
        }
    }

    /// Element-wise multiplication of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply element-wise with this one
    ///
    /// # Returns
    /// A new tensor containing the element-wise product
    ///
    /// # Errors
    /// When shapes do not match or tensors are on different devices
    fn elementwise_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for multiplication"));
        }

        if !self.has_same_device(other) {
            return Err(TensorError::new(
                "Tensors must be on the same device. Use .to(device) to move tensors."
            ));
        }

        match self.device() {
            Device::CPU => self.cpu_mul(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_mul(other),
            #[cfg(not(feature = "cuda"))] Device::CUDA(_) => {
                Err(TensorError::new("CUDA support not compiled. Rebuild with --features cuda"))
            }
        }
    }

    /// In-place element-wise multiplication of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply element-wise with this one
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match or tensors are on different devices
    fn elementwise_mul_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for multiplication"));
        }

        if !self.has_same_device(other) {
            return Err(TensorError::new(
                "Tensors must be on the same device. Use .to(device) to move tensors."
            ));
        }

        match self.device() {
            Device::CPU => self.cpu_mul_mut(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_mul_mut(other),
            #[cfg(not(feature = "cuda"))] Device::CUDA(_) => {
                Err(TensorError::new("CUDA support not compiled. Rebuild with --features cuda"))
            }
        }
    }

    /// Element-wise division of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to divide element-wise with this one
    ///
    /// # Returns
    /// A new tensor containing the element-wise quotient
    ///
    /// # Errors
    /// When shapes do not match or tensors are on different devices
    fn elementwise_div(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for division"));
        }

        if !self.has_same_device(other) {
            return Err(TensorError::new(
                "Tensors must be on the same device. Use .to(device) to move tensors."
            ));
        }

        match self.device() {
            Device::CPU => self.cpu_div(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_div(other),
            #[cfg(not(feature = "cuda"))] Device::CUDA(_) => {
                Err(TensorError::new("CUDA support not compiled. Rebuild with --features cuda"))
            }
        }
    }

    /// In-place element-wise division of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to divide element-wise with this one
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match or tensors are on different devices
    fn elementwise_div_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for division"));
        }

        if !self.has_same_device(other) {
            return Err(TensorError::new(
                "Tensors must be on the same device. Use .to(device) to move tensors."
            ));
        }

        match self.device() {
            Device::CPU => self.cpu_div_mut(other),
            #[cfg(feature = "cuda")]
            Device::CUDA(_) => self.cuda_div_mut(other),
            #[cfg(not(feature = "cuda"))] Device::CUDA(_) => {
                Err(TensorError::new("CUDA support not compiled. Rebuild with --features cuda"))
            }
        }
    }
}

// CPU implementations
impl<T: TensorNum> Tensor<T> {
    fn cpu_add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for addition"));
        }

        match (&self.storage, &other.storage) {
            (TensorStorage::CPU(self_data), TensorStorage::CPU(other_data)) => {
                let data = self_data
                    .iter()
                    .zip(other_data.iter())
                    .map(|(a, b)| *a + *b)
                    .collect();

                Ok(Tensor {
                    storage: TensorStorage::CPU(data),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
        }
    }

    fn cpu_add_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        match (&mut self.storage, &other.storage) {
            (TensorStorage::CPU(self_data), TensorStorage::CPU(other_data)) => {
                for (a, b) in self_data.iter_mut().zip(other_data.iter()) {
                    *a = *a + *b;
                }
                Ok(())
            }
        }
    }

    fn cpu_sub(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        match (&self.storage, &other.storage) {
            (TensorStorage::CPU(self_data), TensorStorage::CPU(other_data)) => {
                let data: Vec<T> = self_data
                    .iter()
                    .zip(other_data.iter())
                    .map(|(a, b)| *a - *b)
                    .collect();

                Ok(Tensor {
                    storage: TensorStorage::CPU(data),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            #[cfg(feature = "cuda")]
            _ => Err(TensorError::new("Expected CPU tensor")),
        }
    }

    fn cpu_sub_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        match (&mut self.storage, &other.storage) {
            (TensorStorage::CPU(self_data), TensorStorage::CPU(other_data)) => {
                for (a, b) in self_data.iter_mut().zip(other_data.iter()) {
                    *a = *a - *b;
                }
                Ok(())
            }
            #[cfg(feature = "cuda")]
            _ => Err(TensorError::new("Device mismatch")),
        }
    }

    fn cpu_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        match (&self.storage, &other.storage) {
            (TensorStorage::CPU(self_data), TensorStorage::CPU(other_data)) => {
                let data: Vec<T> = self_data
                    .iter()
                    .zip(other_data.iter())
                    .map(|(a, b)| *a * *b)
                    .collect();

                Ok(Tensor {
                    storage: TensorStorage::CPU(data),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            #[cfg(feature = "cuda")]
            _ => Err(TensorError::new("Expected CPU tensor")),
        }
    }

    fn cpu_mul_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        match (&mut self.storage, &other.storage) {
            (TensorStorage::CPU(self_data), TensorStorage::CPU(other_data)) => {
                for (a, b) in self_data.iter_mut().zip(other_data.iter()) {
                    *a = *a * *b;
                }
                Ok(())
            }
            #[cfg(feature = "cuda")]
            _ => Err(TensorError::new("Device mismatch")),
        }
    }

    fn cpu_div(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        match (&self.storage, &other.storage) {
            (TensorStorage::CPU(self_data), TensorStorage::CPU(other_data)) => {
                let data: Vec<T> = self_data
                    .iter()
                    .zip(other_data.iter())
                    .map(|(a, b)| *a / *b)
                    .collect();

                Ok(Tensor {
                    storage: TensorStorage::CPU(data),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            #[cfg(feature = "cuda")]
            _ => Err(TensorError::new("Expected CPU tensor")),
        }
    }

    fn cpu_div_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        match (&mut self.storage, &other.storage) {
            (TensorStorage::CPU(self_data), TensorStorage::CPU(other_data)) => {
                for (a, b) in self_data.iter_mut().zip(other_data.iter()) {
                    *a = *a / *b;
                }
                Ok(())
            }
            #[cfg(feature = "cuda")]
            _ => Err(TensorError::new("Device mismatch")),
        }
    }
}

// CUDA implementations for f32
#[cfg(feature = "cuda")]
impl Tensor<f32> {
    fn cuda_add(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match (&self.storage, &other.storage) {
            (
                TensorStorage::CUDA { context, buffer: a_buf, device_idx },
                TensorStorage::CUDA { buffer: b_buf, .. }
            ) => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut c_buf = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::add_f32(&backend, a_buf, b_buf, &mut c_buf, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: c_buf,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensors")),
        }
    }

    fn cuda_add_mut(&mut self, other: &Tensor<f32>) -> Result<(), TensorError> {
        // For now, implement via add + replace
        // TODO: Optimize with in-place kernel
        let result = self.cuda_add(other)?;
        self.storage = result.storage;
        Ok(())
    }

    fn cuda_sub(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match (&self.storage, &other.storage) {
            (
                TensorStorage::CUDA { context, buffer: a_buf, device_idx },
                TensorStorage::CUDA { buffer: b_buf, .. }
            ) => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut c_buf = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::sub_f32(&backend, a_buf, b_buf, &mut c_buf, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: c_buf,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensors")),
        }
    }

    fn cuda_sub_mut(&mut self, other: &Tensor<f32>) -> Result<(), TensorError> {
        let result = self.cuda_sub(other)?;
        self.storage = result.storage;
        Ok(())
    }

    fn cuda_mul(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match (&self.storage, &other.storage) {
            (
                TensorStorage::CUDA { context, buffer: a_buf, device_idx },
                TensorStorage::CUDA { buffer: b_buf, .. }
            ) => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut c_buf = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::mul_f32(&backend, a_buf, b_buf, &mut c_buf, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: c_buf,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensors")),
        }
    }

    fn cuda_mul_mut(&mut self, other: &Tensor<f32>) -> Result<(), TensorError> {
        let result = self.cuda_mul(other)?;
        self.storage = result.storage;
        Ok(())
    }

    fn cuda_div(&self, other: &Tensor<f32>) -> Result<Tensor<f32>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match (&self.storage, &other.storage) {
            (
                TensorStorage::CUDA { context, buffer: a_buf, device_idx },
                TensorStorage::CUDA { buffer: b_buf, .. }
            ) => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut c_buf = context.alloc_zeros::<f32>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::div_f32(&backend, a_buf, b_buf, &mut c_buf, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: c_buf,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensors")),
        }
    }

    fn cuda_div_mut(&mut self, other: &Tensor<f32>) -> Result<(), TensorError> {
        let result = self.cuda_div(other)?;
        self.storage = result.storage;
        Ok(())
    }
}

// CUDA implementations for f64
#[cfg(feature = "cuda")]
impl Tensor<f64> {
    fn cuda_add(&self, other: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match (&self.storage, &other.storage) {
            (
                TensorStorage::CUDA { context, buffer: a_buf, device_idx },
                TensorStorage::CUDA { buffer: b_buf, .. }
            ) => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut c_buf = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::add_f64(&backend, a_buf, b_buf, &mut c_buf, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: c_buf,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensors")),
        }
    }

    fn cuda_add_mut(&mut self, other: &Tensor<f64>) -> Result<(), TensorError> {
        let result = self.cuda_add(other)?;
        self.storage = result.storage;
        Ok(())
    }

    fn cuda_sub(&self, other: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match (&self.storage, &other.storage) {
            (
                TensorStorage::CUDA { context, buffer: a_buf, device_idx },
                TensorStorage::CUDA { buffer: b_buf, .. }
            ) => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut c_buf = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::sub_f64(&backend, a_buf, b_buf, &mut c_buf, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: c_buf,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensors")),
        }
    }

    fn cuda_sub_mut(&mut self, other: &Tensor<f64>) -> Result<(), TensorError> {
        let result = self.cuda_sub(other)?;
        self.storage = result.storage;
        Ok(())
    }

    fn cuda_mul(&self, other: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match (&self.storage, &other.storage) {
            (
                TensorStorage::CUDA { context, buffer: a_buf, device_idx },
                TensorStorage::CUDA { buffer: b_buf, .. }
            ) => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut c_buf = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::mul_f64(&backend, a_buf, b_buf, &mut c_buf, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: c_buf,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensors")),
        }
    }

    fn cuda_mul_mut(&mut self, other: &Tensor<f64>) -> Result<(), TensorError> {
        let result = self.cuda_mul(other)?;
        self.storage = result.storage;
        Ok(())
    }

    fn cuda_div(&self, other: &Tensor<f64>) -> Result<Tensor<f64>, TensorError> {
        use corrosive_cuda::{CudaBackend, kernels::ElementwiseKernels};

        match (&self.storage, &other.storage) {
            (
                TensorStorage::CUDA { context, buffer: a_buf, device_idx },
                TensorStorage::CUDA { buffer: b_buf, .. }
            ) => {
                let backend = CudaBackend::new(*device_idx)
                    .map_err(|e| TensorError::new(&format!("CUDA backend error: {}", e)))?;

                let n = self.size();
                let mut c_buf = context.alloc_zeros::<f64>(n)
                    .map_err(|e| TensorError::new(&format!("CUDA alloc failed: {}", e)))?;

                ElementwiseKernels::div_f64(&backend, a_buf, b_buf, &mut c_buf, n)
                    .map_err(|e| TensorError::new(&format!("CUDA kernel launch failed: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: context.clone(),
                        buffer: c_buf,
                        device_idx: *device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            _ => Err(TensorError::new("Expected CUDA tensors")),
        }
    }

    fn cuda_div_mut(&mut self, other: &Tensor<f64>) -> Result<(), TensorError> {
        let result = self.cuda_div(other)?;
        self.storage = result.storage;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Device, Tensor, TensorArithmetic, TensorCore, TensorInit};

    #[test]
    fn test_element_wise_addition() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU).unwrap();

        let result = a.add(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 6.0);  // 1 + 5
        assert_eq!(*result.get(&[0, 1]).unwrap(), 8.0);  // 2 + 6
        assert_eq!(*result.get(&[1, 0]).unwrap(), 10.0); // 3 + 7
        assert_eq!(*result.get(&[1, 1]).unwrap(), 12.0); // 4 + 8
    }

    #[test]
    fn test_element_wise_addition_mut() {
        let mut a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU).unwrap();

        a.add_mut(&b).unwrap();
        assert_eq!(*a.get(&[0, 0]).unwrap(), 6.0);
        assert_eq!(*a.get(&[1, 1]).unwrap(), 12.0);
    }

    #[test]
    fn test_element_wise_subtraction() {
        let a = Tensor::<f32>::from_data(vec![10.0, 8.0, 6.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();

        let result = a.sub(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 9.0);  // 10 - 1
        assert_eq!(*result.get(&[0, 1]).unwrap(), 6.0);  // 8 - 2
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0);  // 6 - 3
        assert_eq!(*result.get(&[1, 1]).unwrap(), 0.0);  // 4 - 4
    }

    #[test]
    fn test_elementwise_multiplication() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], Device::CPU).unwrap();

        let result = a.elementwise_mul(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0);  // 1 * 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 6.0);  // 2 * 3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 12.0); // 3 * 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 20.0); // 4 * 5
    }

    #[test]
    fn test_elementwise_division() {
        let a = Tensor::<f32>::from_data(vec![8.0, 12.0, 16.0, 20.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], Device::CPU).unwrap();

        let result = a.elementwise_div(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 4.0);  // 8 / 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);  // 12 / 3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 4.0);  // 16 / 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0);  // 20 / 5
    }

    #[test]
    fn test_error_handling_shape_mismatch() {
        let a = Tensor::<f32>::zeros(vec![2, 3], Device::CPU).unwrap();
        let b = Tensor::<f32>::zeros(vec![3, 2], Device::CPU).unwrap();

        let result = a.add(&b);
        assert!(result.is_err());

        let mut a_mut = Tensor::<f32>::zeros(vec![2, 3], Device::CPU).unwrap();
        let result = a_mut.add_mut(&b);
        assert!(result.is_err());
    }
}
