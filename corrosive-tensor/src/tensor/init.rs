use super::{Tensor, TensorError, TensorCore, TensorNum, TensorStorage, Device};

pub trait TensorInit<T> {
    fn zeros(shape: Vec<usize>, device: Device) -> Result<Self, TensorError> where Self: Sized;
    fn ones(shape: Vec<usize>, device: Device) -> Result<Self, TensorError> where Self: Sized;
    fn from_data(data: Vec<T>, shape: Vec<usize>, device: Device) -> Result<Self, TensorError> where Self: Sized;
}

impl<T> TensorInit<T> for Tensor<T>
where
    T: TensorNum,
{
    /// Create a new tensor filled with zeros.
    ///
    /// # Arguments
    /// * `shape` - The dimensions of the tensor
    /// * `device` - The device where the tensor will be allocated (CPU or CUDA)
    ///
    /// # Returns
    /// A new tensor with all elements set to zero
    fn zeros(shape: Vec<usize>, device: Device) -> Result<Self, TensorError> {
        Self::validation(&shape)?;

        let size: usize = shape.iter().product();
        let strides = Self::calculate_strides(&shape);

        match device {
            Device::CPU => Ok(Tensor {
                storage: TensorStorage::CPU(vec![T::default(); size]),
                shape,
                strides,
            }),
            #[cfg(feature = "cuda")]
            Device::CUDA(device_idx) => {
                use crate::cuda::CudaBackend;

                let backend = CudaBackend::new(device_idx)
                    .map_err(|e| TensorError::new(&format!("Failed to initialize CUDA: {}", e)))?;

                let stream = backend.context().default_stream();
                let buffer = stream.alloc_zeros::<T>(size)
                    .map_err(|e| TensorError::new(&format!("Failed to allocate CUDA memory: {}", e)))?;

                Ok(Tensor {
                    storage: TensorStorage::CUDA {
                        context: backend.context().clone(),
                        buffer,
                        device_idx,
                    },
                    shape,
                    strides,
                })
            }
            #[cfg(not(feature = "cuda"))]
            Device::CUDA(_) => {
                Err(TensorError::new(
                    "CUDA support not compiled. Rebuild with --features cuda"
                ))
            }
        }
    }

    /// Create a new tensor filled with ones.
    ///
    /// # Arguments
    /// * `shape` - The dimensions of the tensor
    /// * `device` - The device where the tensor will be allocated (CPU or CUDA)
    ///
    /// # Returns
    /// A new tensor with all elements set to one
    ///
    /// # Errors
    /// When shape contains zero dimensions
    fn ones(shape: Vec<usize>, device: Device) -> Result<Self, TensorError> {
        Self::validation(&shape)?;

        let size: usize = shape.iter().product();
        let strides = Self::calculate_strides(&shape);

        // For now, create on CPU and transfer to CUDA if needed
        // TODO: Optimize to directly create ones on CUDA
        let cpu_tensor = Tensor {
            storage: TensorStorage::CPU(vec![T::one(); size]),
            shape,
            strides,
        };

        match device {
            Device::CPU => Ok(cpu_tensor),
            cuda_device => cpu_tensor.to(cuda_device),
        }
    }

    /// Create a tensor from existing data with the specified shape.
    ///
    /// # Arguments
    /// * `data` - Vector containing the tensor data in row-major order
    /// * `shape` - The dimensions of the tensor
    /// * `device` - The device where the tensor will be allocated (CPU or CUDA)
    ///
    /// # Returns
    /// A new tensor containing the provided data
    ///
    /// # Errors
    /// When data length does not match shape or shape contains zero dimensions
    fn from_data(data: Vec<T>, shape: Vec<usize>, device: Device) -> Result<Self, TensorError> {
        let expected_size: usize = shape.iter().product();

        if data.len() != expected_size {
            return Err(TensorError::new("Data size does not match shape"));
        }

        Self::validation(&shape)?;

        let strides = Self::calculate_strides(&shape);
        let cpu_tensor = Tensor {
            storage: TensorStorage::CPU(data),
            shape,
            strides,
        };

        match device {
            Device::CPU => Ok(cpu_tensor),
            cuda_device => cpu_tensor.to(cuda_device),
        }
    }
}

impl<T> Tensor<T> {
    fn validation(shape: &[usize]) -> Result<(), TensorError> {
        if shape.is_empty() {
            return Err(TensorError::new("Tensor must have at least one dimension"));
        }

        for &dim in shape {
            if dim == 0 {
                return Err(TensorError::new("Tensor dimensions must be greater than 0"));
            }
        }

        Ok(())
    }
}