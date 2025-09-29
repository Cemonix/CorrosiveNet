use super::{Tensor, TensorError, TensorCore, TensorStorage, TensorNum};

pub trait TensorLinearAlgebra<T> {
    fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn broadcast_add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn can_broadcast_with(&self, other: &Tensor<T>) -> bool;
}

impl<T> TensorLinearAlgebra<T> for Tensor<T>
where
    T: TensorNum,
{
    /// Perform tensor multiplication between two 2D tensors.
    ///
    /// Computes the tensor product C = A × B where:
    /// - A is an m×k tensor (self)
    /// - B is a k×n tensor (other)
    /// - C is an m×n tensor (result)
    ///
    /// # Arguments
    /// * `other` - The right-hand tensor to multiply with
    ///
    /// # Returns
    /// A new tensor containing the tensor product
    ///
    /// # Errors
    /// Returns an error if:
    /// - Either tensor is not 2D
    /// - The inner dimensions don't match (A.cols ≠ B.rows)
    fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        if self.shape().len() != 2 || other.shape().len() != 2 {
            return Err(TensorError::new(&format!(
                "tensors must be 2D, got {}D and {}D", self.shape().len(), other.shape().len()
            )));
        }

        if self.shape()[1] != other.shape()[0] {
            return Err(TensorError::new(&format!(
                "Tensor dimension mismatch: cannot multiply {}×{} with {}×{} (inner dimensions {} ≠ {})",
                self.shape()[0], self.shape()[1], other.shape()[0], other.shape()[1],
                self.shape()[1], other.shape()[0]
            )));
        }

        let rows = self.shape()[0];
        let cols = other.shape()[1];
        let inner = self.shape()[1];

        let result_shape = vec![rows, cols];
        let mut result = Tensor::zeros(result_shape)?;

        let self_data = &self.data;
        let other_data = &other.data;
        let result_strides = result.strides().to_vec();
        let result_data = &mut result.data;

        for i in 0..rows {
            for j in 0..cols {
                let mut sum = T::zero();

                for k in 0..inner {
                    let self_idx = i * self.strides()[0] + k * self.strides()[1];
                    let other_idx = k * other.strides()[0] + j * other.strides()[1];
                    sum = sum + self_data[self_idx] * other_data[other_idx];
                }

                let result_idx = i * result_strides[0] + j * result_strides[1];
                result_data[result_idx] = sum;
            }
        }

        Ok(result)
    }

    /// Check if two tensors can be broadcasted together.
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
        let mut result = Tensor::zeros(result_shape.clone())?;

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
            result.set(&indices, self_val + other_val)?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::tensor::{Tensor, TensorCore, TensorStorage, TensorLinearAlgebra};

    #[test]
    fn test_tensor_multiplication_basic() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::<f32>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        let result = a.matmul(&b).unwrap();

        assert_eq!(*result.get(&[0, 0]).unwrap(), 19.0);  // 1*5 + 2*7 = 19
        assert_eq!(*result.get(&[0, 1]).unwrap(), 22.0);  // 1*6 + 2*8 = 22
        assert_eq!(*result.get(&[1, 0]).unwrap(), 43.0);  // 3*5 + 4*7 = 43
        assert_eq!(*result.get(&[1, 1]).unwrap(), 50.0);  // 3*6 + 4*8 = 50
    }

    #[test]
    fn test_tensor_multiplication_rectangular() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::<f32>::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 58.0);   // 1*7 + 2*9 + 3*11 = 58
        assert_eq!(*result.get(&[0, 1]).unwrap(), 64.0);   // 1*8 + 2*10 + 3*12 = 64
        assert_eq!(*result.get(&[1, 0]).unwrap(), 139.0);  // 4*7 + 5*9 + 6*11 = 139
        assert_eq!(*result.get(&[1, 1]).unwrap(), 154.0);  // 4*8 + 5*10 + 6*12 = 154
    }

    #[test]
    fn test_error_handling_tensor_multiplication() {
        let a = Tensor::<f32>::zeros(vec![2, 3]).unwrap();
        let b = Tensor::<f32>::zeros(vec![2, 2]).unwrap();

        let result = a.matmul(&b);
        assert!(result.is_err());

        let a_3d = Tensor::<f32>::zeros(vec![2, 3, 4]).unwrap();
        let b_3d = Tensor::<f32>::zeros(vec![2, 3, 4]).unwrap();
        let result = a_3d.matmul(&b_3d);
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcasting_compatibility() {
        let a = Tensor::<f32>::zeros(vec![3, 1]).unwrap();
        let b = Tensor::<f32>::zeros(vec![1, 4]).unwrap();
        assert!(a.can_broadcast_with(&b));

        let c = Tensor::<f32>::zeros(vec![3, 2]).unwrap();
        let d = Tensor::<f32>::zeros(vec![3, 4]).unwrap();
        assert!(!c.can_broadcast_with(&d));
    }
}