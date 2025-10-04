use super::{Tensor, TensorError, TensorNum, TensorCore, TensorStorage};

pub trait TensorBroadcast<T> {
    fn can_broadcast_with(&self, other: &Tensor<T>) -> bool;
    fn broadcast_add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn broadcast_sub(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn broadcast_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn broadcast_div(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
}

impl<T> TensorBroadcast<T> for Tensor<T>
where
    T: TensorNum,
{
    /// Check if two tensors can be broadcasted together.
    ///
    /// Broadcasting rules:
    /// - Dimensions are compared from right to left
    /// - Each dimension pair must be either equal or one of them is 1
    /// - Missing dimensions are treated as 1
    ///
    /// # Arguments
    /// * `other` - The tensor to check broadcasting compatibility with
    ///
    /// # Returns
    /// `true` if tensors can be broadcasted, `false` otherwise
    fn can_broadcast_with(&self, other: &Tensor<T>) -> bool {
        let self_shape = self.shape();
        let other_shape = other.shape();

        let max_dims = self_shape.len().max(other_shape.len());

        for i in 0..max_dims {
            let self_dim = if i < self_shape.len() {
                self_shape[self_shape.len() - 1 - i]
            } else {
                1
            };

            let other_dim = if i < other_shape.len() {
                other_shape[other_shape.len() - 1 - i]
            } else {
                1
            };

            if self_dim != other_dim && self_dim != 1 && other_dim != 1 {
                return false;
            }
        }

        true
    }

    /// Perform element-wise addition with broadcasting.
    ///
    /// # Arguments
    /// * `other` - The tensor to add with broadcasting
    ///
    /// # Returns
    /// A new tensor containing the broadcasted sum
    ///
    /// # Errors
    /// When tensors cannot be broadcasted together
    fn broadcast_add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        self.broadcast_op(other, |a, b| a + b)
    }

    /// Perform element-wise subtraction with broadcasting.
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract with broadcasting
    ///
    /// # Returns
    /// A new tensor containing the broadcasted difference
    ///
    /// # Errors
    /// When tensors cannot be broadcasted together
    fn broadcast_sub(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        self.broadcast_op(other, |a, b| a - b)
    }

    /// Perform element-wise multiplication with broadcasting.
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply with broadcasting
    ///
    /// # Returns
    /// A new tensor containing the broadcasted product
    ///
    /// # Errors
    /// When tensors cannot be broadcasted together
    fn broadcast_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        self.broadcast_op(other, |a, b| a * b)
    }

    /// Perform element-wise division with broadcasting.
    ///
    /// # Arguments
    /// * `other` - The tensor to divide with broadcasting
    ///
    /// # Returns
    /// A new tensor containing the broadcasted quotient
    ///
    /// # Errors
    /// When tensors cannot be broadcasted together
    fn broadcast_div(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        self.broadcast_op(other, |a, b| a / b)
    }
}

impl<T> Tensor<T> {
    /// Generic helper for broadcasting operations between two tensors.
    ///
    /// # Arguments
    /// * `other` - The other tensor to operate with
    /// * `op` - The binary operation to apply element-wise
    ///
    /// # Returns
    /// A new tensor containing the result of the broadcasted operation
    ///
    /// # Errors
    /// When tensors cannot be broadcasted together
    pub(super) fn broadcast_op<F>(&self, other: &Tensor<T>, op: F) -> Result<Tensor<T>, TensorError>
    where
        F: Fn(T, T) -> T,
        T: TensorNum,
    {
        if !self.has_same_device(other) {
            return Err(TensorError::new("Tensors are on different devices"));
        }

        if !self.can_broadcast_with(other) {
            return Err(TensorError::new(&format!(
                "Cannot broadcast shapes {:?} and {:?}",
                self.shape(),
                other.shape()
            )));
        }

        let self_shape = self.shape();
        let other_shape = other.shape();

        let max_dims = self_shape.len().max(other_shape.len());
        let mut result_shape = Vec::with_capacity(max_dims);

        for i in 0..max_dims {
            let self_dim = if i < self_shape.len() {
                self_shape[self_shape.len() - 1 - i]
            } else {
                1
            };

            let other_dim = if i < other_shape.len() {
                other_shape[other_shape.len() - 1 - i]
            } else {
                1
            };

            result_shape.push(self_dim.max(other_dim));
        }

        result_shape.reverse();
        let mut result = Tensor::zeros(result_shape.clone(), self.device.clone())?;

        let total_elements = result.size();
        for i in 0..total_elements {
            let mut indices = Vec::new();
            let mut temp_i = i;
            for &dim in result_shape.iter().rev() {
                indices.push(temp_i % dim);
                temp_i /= dim;
            }
            indices.reverse();

            let mut self_indices = Vec::new();
            let mut other_indices = Vec::new();

            for (idx, &result_idx) in indices.iter().enumerate() {
                if idx < self_shape.len() {
                    let self_dim = self_shape[idx];
                    self_indices.push(if self_dim == 1 { 0 } else { result_idx });
                } else {
                    self_indices.push(0);
                }

                if idx < other_shape.len() {
                    let other_dim = other_shape[idx];
                    other_indices.push(if other_dim == 1 { 0 } else { result_idx });
                } else {
                    other_indices.push(0);
                }
            }

            self_indices.truncate(self_shape.len());
            other_indices.truncate(other_shape.len());

            let self_val = *self.get(&self_indices)?;
            let other_val = *other.get(&other_indices)?;
            result.set(&indices, op(self_val, other_val))?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Device, Tensor, TensorBroadcast, TensorCore, TensorStorage};

    #[test]
    fn test_broadcasting_compatibility() {
        let a = Tensor::<f32>::zeros(vec![3, 1], Device::CPU).unwrap();
        let b = Tensor::<f32>::zeros(vec![1, 4], Device::CPU).unwrap();
        assert!(a.can_broadcast_with(&b));

        let c = Tensor::<f32>::zeros(vec![3, 2], Device::CPU).unwrap();
        let d = Tensor::<f32>::zeros(vec![3, 4], Device::CPU).unwrap();
        assert!(!c.can_broadcast_with(&d));
    }

    #[test]
    fn test_broadcast_add_simple() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0], vec![3], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![10.0], vec![1], Device::CPU).unwrap();

        let result = a.broadcast_add(&b).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(*result.get(&[0]).unwrap(), 11.0);
        assert_eq!(*result.get(&[1]).unwrap(), 12.0);
        assert_eq!(*result.get(&[2]).unwrap(), 13.0);
    }

    #[test]
    fn test_broadcast_add_2d() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![10.0, 20.0], vec![1, 2], Device::CPU).unwrap();

        let result = a.broadcast_add(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 11.0); // 1 + 10
        assert_eq!(*result.get(&[0, 1]).unwrap(), 22.0); // 2 + 20
        assert_eq!(*result.get(&[1, 0]).unwrap(), 13.0); // 3 + 10
        assert_eq!(*result.get(&[1, 1]).unwrap(), 24.0); // 4 + 20
    }

    #[test]
    fn test_broadcast_mul() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![2.0], vec![1], Device::CPU).unwrap();

        let result = a.broadcast_mul(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(*result.get(&[1, 0]).unwrap(), 6.0);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 8.0);
    }

    #[test]
    fn test_broadcast_error() {
        let a = Tensor::<f32>::zeros(vec![3, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::zeros(vec![3, 4], Device::CPU).unwrap();

        let result = a.broadcast_add(&b);
        assert!(result.is_err());
    }
}