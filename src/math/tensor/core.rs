use super::{Tensor, TensorError};

pub trait TensorCore<T> {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn size(&self) -> usize;
    fn get(&self, indices: &[usize]) -> Result<&T, TensorError>;
    fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T, TensorError>;
    fn set(&mut self, indices: &[usize], value: T) -> Result<(), TensorError>;
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
        self.data.len()
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
        Ok(&self.data[flat_index])
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
        Ok(&mut self.data[flat_index])
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
    pub(super) fn index_to_indices(&self, flat_index: usize) -> Vec<usize> {
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