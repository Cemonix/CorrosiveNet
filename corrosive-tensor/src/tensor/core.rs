use crate::{Device, tensor::TensorStorage};

use super::{Tensor, TensorError, TensorElement, TensorNum};

pub trait TensorCore<T> {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn size(&self) -> usize;
    fn device(&self) -> Device;
    fn has_same_device(&self, other: &Tensor<T>) -> bool;
    fn to_vec(&self) -> Result<Vec<T>, TensorError> where T: Clone;
    fn cpu(&self) -> Result<Self, TensorError> where T: Clone, Self: Sized;
    #[cfg(feature = "cuda")]
    fn cuda(&self) -> Result<Self, TensorError> where T: Clone, Self: Sized;
    #[cfg(not(feature = "cuda"))]
    fn cuda(&self) -> Result<Self, TensorError> where T: Clone, Self: Sized;
    fn to(&self, device: Device) -> Result<Self, TensorError> where T: Clone, Self: Sized;
    fn get(&self, indices: &[usize]) -> Result<&T, TensorError>;
    fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T, TensorError>;
    fn set(&mut self, indices: &[usize], value: T) -> Result<(), TensorError>;
    fn fill(&mut self, value: T) where T: TensorElement;
    fn fill_zeros(&mut self) where T: TensorElement;
    fn fill_ones(&mut self) where T: TensorNum;
}

impl<T> TensorCore<T> for Tensor<T> {
    /// Get the shape (dimensions) of the tensor.
    ///
    /// # Returns
    /// Slice containing the size of each dimension
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the tensor.
    ///
    /// # Returns
    /// Slice containing the stride for each dimension
    fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the total number of elements in the tensor.
    ///
    /// # Returns
    /// Total number of elements across all dimensions
    fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Convert tensor data to a Vec (copies to CPU if needed)
    ///
    /// # Returns
    /// Vector containing a copy of the tensor data on CPU
    ///
    /// # Errors
    /// When CUDA memory transfer fails
    fn to_vec(&self) -> Result<Vec<T>, TensorError>
    where
        T: Clone,
    {
        match &self.storage {
            TensorStorage::CPU(data) => Ok(data.clone()),
            #[cfg(feature = "cuda")]
            TensorStorage::CUDA { buffer, .. } => {
                buffer.dtoh_sync_copy()
                    .map_err(|e| TensorError::new(&format!("CUDA memory transfer failed: {}", e)))
            }
        }
    }

    /// Get the device where this tensor's data is stored
    ///
    /// # Returns
    /// The device (CPU or CUDA with device index)
    fn device(&self) -> Device {
        match &self.storage {
            TensorStorage::CPU(_) => Device::CPU,
            #[cfg(feature = "cuda")]
            TensorStorage::CUDA { device_idx, .. } => Device::CUDA(*device_idx),
        }
    }

    /// Check if two tensors are on the same device
    ///
    /// # Arguments
    /// * `other` - The tensor to compare with
    ///
    /// # Returns
    /// True if both tensors are on the same device
    fn has_same_device(&self, other: &Tensor<T>) -> bool {
        match (self.device(), other.device()) {
            (Device::CPU, Device::CPU) => true,
            (Device::CUDA(idx1), Device::CUDA(idx2)) => idx1 == idx2,
            _ => false,
        }
    }

    /// Move tensor to CPU (returns new tensor)
    ///
    /// # Returns
    /// New tensor with data on CPU
    ///
    /// # Errors
    /// When CUDA memory transfer fails
    fn cpu(&self) -> Result<Self, TensorError>
    where
        T: Clone,
    {
        self.to(Device::CPU)
    }

    /// Move tensor to CUDA device (returns new tensor)
    ///
    /// # Returns
    /// New tensor with data on GPU
    ///
    /// # Errors
    /// When CUDA is not available or memory transfer fails
    #[cfg(feature = "cuda")]
    fn cuda(&self) -> Result<Self, TensorError>
    where
        T: Clone,
    {
        self.to(Device::cuda())
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda(&self) -> Result<Self, TensorError>
    where
        T: Clone,
    {
        Err(TensorError::new(
            "CUDA support not compiled. Rebuild with --features cuda"
        ))
    }

    /// Move tensor to specified device (returns new tensor)
    ///
    /// # Arguments
    /// * `device` - Target device (CPU or CUDA)
    ///
    /// # Returns
    /// New tensor on target device (original unchanged)
    ///
    /// # Errors
    /// When device transfer fails
    fn to(&self, device: Device) -> Result<Self, TensorError>
    where
        T: Clone,
    {
        // Already on target device - just clone
        if self.device() == device {
            return Ok(self.clone());
        }

        match (&self.storage, device) {
            // CPU to CUDA transfer
            #[cfg(feature = "cuda")]
            (TensorStorage::CPU(data), Device::CUDA(device_idx)) => {
                use corrosive_cuda::CudaBackend;

                let backend = CudaBackend::new(device_idx)
                    .map_err(|e| TensorError::new(&format!("Failed to initialize CUDA: {}", e)))?;

                let buffer = backend.context().htod_sync_copy(data)
                    .map_err(|e| TensorError::new(&format!("Failed to copy to CUDA: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: backend.context().clone(),
                        buffer,
                        device_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            // CUDA to CPU transfer
            #[cfg(feature = "cuda")]
            (TensorStorage::CUDA { buffer, .. }, Device::CPU) => {
                let data = buffer.dtoh_sync_copy()
                    .map_err(|e| TensorError::new(&format!("Failed to copy from CUDA: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CPU(data),
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            // CUDA to different CUDA device
            #[cfg(feature = "cuda")]
            (TensorStorage::CUDA { .. }, Device::CUDA(target_idx)) => {
                use corrosive_cuda::CudaBackend;

                // Transfer via CPU (peer-to-peer transfer is more complex)
                let data = self.to_vec()?;
                let backend = CudaBackend::new(target_idx)
                    .map_err(|e| TensorError::new(&format!("Failed to initialize CUDA: {}", e)))?;

                let new_buffer = backend.context().htod_sync_copy(&data)
                    .map_err(|e| TensorError::new(&format!("Failed to copy to CUDA: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: backend.context().clone(),
                        buffer: new_buffer,
                        device_idx: target_idx,
                    },
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                })
            }
            #[cfg(not(feature = "cuda"))]
            (_, Device::CUDA(_)) => {
                Err(TensorError::new(
                    "CUDA support not compiled. Rebuild with --features cuda"
                ))
            }
            (_, Device::CPU) => {
                // This should be unreachable due to the early return above
                Ok(self.clone())
            }
        }
    }

    /// Get a reference to an element at the specified indices.
    ///
    /// # Arguments
    /// * `indices` - Array of indices for each dimension
    ///
    /// # Returns
    /// Reference to the element at the specified position
    ///
    /// # Errors
    /// When indices are out of bounds or incorrect number of indices provided
    fn get(&self, indices: &[usize]) -> Result<&T, TensorError> {
        let flat_index = self.index(indices)?;
        match &self.storage {
            TensorStorage::CPU(data) => Ok(&data[flat_index]),
            #[cfg(feature = "cuda")]
            TensorStorage::CUDA { .. } => {
                Err(TensorError::new(
                    "Cannot directly access elements of CUDA tensor. Use .cpu() or .to_vec() first."
                ))
            }
        }
    }

    /// Get a mutable reference to an element at the specified indices.
    ///
    /// # Arguments
    /// * `indices` - Array of indices for each dimension
    ///
    /// # Returns
    /// Mutable reference to the element at the specified position
    ///
    /// # Errors
    /// When indices are out of bounds or incorrect number of indices provided
    fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T, TensorError> {
        let flat_index = self.index(indices)?;
        match &mut self.storage {
            TensorStorage::CPU(data) => Ok(&mut data[flat_index]),
            #[cfg(feature = "cuda")]
            TensorStorage::CUDA { .. } => {
                Err(TensorError::new(
                    "Cannot directly access elements of CUDA tensor. Use .cpu() first."
                ))
            }
        }
    }

    /// Set the value of an element at the specified indices.
    ///
    /// # Arguments
    /// * `indices` - Array of indices for each dimension
    /// * `value` - The value to set
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When indices are out of bounds or incorrect number of indices provided
    fn set(&mut self, indices: &[usize], value: T) -> Result<(), TensorError> {
        let slot = self.get_mut(indices)?;
        *slot = value;
        Ok(())
    }

    fn fill(&mut self, value: T)
        where T: TensorElement
    {
        match &mut self.storage {
            TensorStorage::CPU(data) => {
                for elem in data {
                    *elem = value;
                }
            }
            #[cfg(feature = "cuda")]
            TensorStorage::CUDA { .. } => {
                panic!("fill() not yet implemented for CUDA tensors. Use .cpu() first or wait for CUDA kernel implementation.");
            }
        }
    }

    fn fill_zeros(&mut self)
        where T: TensorElement
    {
        match &mut self.storage {
            TensorStorage::CPU(data) => {
                let zero = T::default();
                for elem in data {
                    *elem = zero;
                }
            }
            #[cfg(feature = "cuda")]
            TensorStorage::CUDA { .. } => {
                panic!("fill_zeros() not yet implemented for CUDA tensors. Use .cpu() first or wait for CUDA kernel implementation.");
            }
        }
    }

    fn fill_ones(&mut self)
        where T: TensorNum
    {
        match &mut self.storage {
            TensorStorage::CPU(data) => {
                let one = T::one();
                for elem in data {
                    *elem = one;
                }
            }
            #[cfg(feature = "cuda")]
            TensorStorage::CUDA { .. } => {
                panic!("fill_ones() not yet implemented for CUDA tensors. Use .cpu() first or wait for CUDA kernel implementation.");
            }
        }
    }
}

impl<T> Tensor<T> {
    /// Calculate flat index from multi-dimensional indices.
    ///
    /// This is a private helper method used internally by the core trait methods.
    /// It's not exposed in the public API since users should work with multi-dimensional
    /// indices rather than flat indices.
    ///
    /// # Arguments
    /// * `indices` - Array of indices for each dimension
    ///
    /// # Returns
    /// The corresponding flat index in the data vector
    ///
    /// # Errors
    /// When indices are out of bounds or incorrect number of indices provided
    pub(super) fn index(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::new("Incorrect number of indices"));
        }
        let mut flat_index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(TensorError::new("Index out of bounds"));
            }
            flat_index += idx * self.strides[i];
        }
        Ok(flat_index)
    }

    /// Convert flat index to multi-dimensional indices.
    /// 
    /// # Arguments
    /// * `flat_index` - The flat index in the data vector
    ///
    /// # Returns
    /// A vector containing the multi-dimensional indices
    pub(super) fn flat_to_indices(&self, flat_index: usize) -> Vec<usize> {
        let n = self.shape.len();
        let mut indices = vec![0; n];
        let mut remaining = flat_index;

        for i in 0..n {
            indices[i] = remaining / self.strides[i];
            remaining %= self.strides[i];
        }

        indices
    }

    /// Calculate strides for row-major order.
    /// 
    /// # Arguments
    /// * `shape` - The dimensions of the tensor
    /// 
    /// # Returns
    /// A vector containing the stride for each dimension
    pub(super) fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let n = shape.len();
        let mut strides = vec![1; n];
        for i in (0..n - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Check if the tensor data is in contiguous memory layout.
    ///
    /// A tensor is contiguous if its strides match the expected row-major strides.
    /// This affects performance - contiguous tensors can use faster operations.
    ///
    /// # Returns
    /// True if tensor is stored contiguously in memory
    pub(super) fn is_contiguous(&self) -> bool {
        let expected_strides = Self::calculate_strides(&self.shape);
        self.strides == expected_strides
    }
}