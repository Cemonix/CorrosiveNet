use super::{Tensor, TensorError, TensorBool};

pub trait TensorMask<T> {
    fn logical_and(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn logical_or(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn logical_not(&self) -> Tensor<T>;
    fn any(&self) -> bool;
    fn all(&self) -> bool;
    fn count_true(&self) -> usize;
}

impl<T> TensorMask<T> for Tensor<T>
where
    T: TensorBool,
{
    /// Element-wise logical AND operation between two boolean tensors.
    ///
    /// # Arguments
    /// * `other` - The other boolean tensor to perform AND with
    ///
    /// # Returns
    /// A new tensor with element-wise AND results
    ///
    /// # Errors
    /// When shapes do not match
    fn logical_and(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        self.elementwise_op(other, |a, b| a.and(b))
    }

    /// Element-wise logical OR operation between two boolean tensors.
    ///
    /// # Arguments
    /// * `other` - The other boolean tensor to perform OR with
    ///
    /// # Returns
    /// A new tensor with element-wise OR results
    ///
    /// # Errors
    /// When shapes do not match
    fn logical_or(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        self.elementwise_op(other, |a, b| a.or(b))
    }

    /// Element-wise logical NOT operation on the tensor.
    ///
    /// # Returns
    /// A new tensor with all boolean values inverted
    fn logical_not(&self) -> Tensor<T> {
        let data: Vec<T> = self.data.iter().map(|&x| x.not()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            device: self.device.clone(),
        }
    }

    /// Check if any element in the tensor is true.
    ///
    /// # Returns
    /// True if at least one element is true, false otherwise
    fn any(&self) -> bool {
        self.data.iter().any(|&x| x.is_true())
    }

    /// Check if all elements in the tensor are true.
    ///
    /// # Returns
    /// True if all elements are true, false otherwise
    fn all(&self) -> bool {
        self.data.iter().all(|&x| x.is_true())
    }

    /// Count the number of true elements in the tensor.
    ///
    /// # Returns
    /// The number of elements that are true
    fn count_true(&self) -> usize {
        self.data.iter().filter(|&&x| x.is_true()).count()
    }
}