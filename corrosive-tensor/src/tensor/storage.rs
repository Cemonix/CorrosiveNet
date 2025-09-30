use super::{Tensor, TensorError, TensorNum};

pub trait TensorStorage<T> {
    fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> where Self: Sized;
    fn ones(shape: Vec<usize>) -> Result<Self, TensorError> where Self: Sized;
    fn from_data(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> where Self: Sized;
}

impl<T> TensorStorage<T> for Tensor<T>
where
    T: TensorNum,
{
    /// Create a new tensor filled with zeros.
    ///
    /// # Arguments
    /// * `shape` - The dimensions of the tensor
    ///
    /// # Returns
    /// A new tensor with all elements set to zero
    fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        Self::validation(&shape)?;

        let size: usize = shape.iter().product();
        let strides = Self::calculate_strides(&shape);

        Ok(Tensor {
            data: vec![T::default(); size],
            shape,
            strides,
        })
    }

    /// Create a new tensor filled with ones.
    ///
    /// # Arguments
    /// * `shape` - The dimensions of the tensor
    ///
    /// # Returns
    /// A new tensor with all elements set to one
    ///
    /// # Errors
    /// When shape contains zero dimensions
    fn ones(shape: Vec<usize>) -> Result<Self, TensorError> {
        Self::validation(&shape)?;

        let size: usize = shape.iter().product();
        let strides = Self::calculate_strides(&shape);

        Ok(Tensor {
            data: vec![T::one(); size],
            shape,
            strides,
        })
    }

    /// Create a tensor from existing data with the specified shape.
    ///
    /// # Arguments
    /// * `data` - Vector containing the tensor data in row-major order
    /// * `shape` - The dimensions of the tensor
    ///
    /// # Returns
    /// A new tensor containing the provided data
    ///
    /// # Errors
    /// When data length does not match shape or shape contains zero dimensions
    fn from_data(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let expected_size: usize = shape.iter().product();

        if data.len() != expected_size {
            return Err(TensorError::new("Data size does not match shape"));
        }

        Self::validation(&shape)?;

        let strides = Self::calculate_strides(&shape);
        Ok(Tensor {
            data,
            shape,
            strides,
        })
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