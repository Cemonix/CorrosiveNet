use super::{Tensor, TensorError, TensorCore, TensorStorage, TensorNum};

pub trait TensorLinAlg<T> {
    fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
}

impl<T> TensorLinAlg<T> for Tensor<T>
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
        if self.has_same_device(other) == false {
            return Err(TensorError::new("tensors are on different devices"));
        }

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
        let mut result = Tensor::zeros(result_shape, self.device.clone())?;

        let self_data = &self.data;
        let other_data = &other.data;
        let self_row_stride = self.strides()[0];
        let self_col_stride = self.strides()[1];
        let other_row_stride = other.strides()[0];
        let other_col_stride = other.strides()[1];
        let result_row_stride = result.strides()[0];
        let result_col_stride = result.strides()[1];
        let result_data = &mut result.data;

        for j in 0..cols {
            let other_col_offset = j * other_col_stride;
            let result_col_offset = j * result_col_stride;

            for i in 0..rows {
                let self_row_offset = i * self_row_stride;
                let result_idx = i * result_row_stride + result_col_offset;
                let mut sum = T::zero();

                for k in 0..inner {
                    let a_ik = self_data[self_row_offset + k * self_col_stride];
                    let b_kj = other_data[k * other_row_stride + other_col_offset];
                    sum = sum + a_ik * b_kj;
                }

                result_data[result_idx] = sum;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Device, Tensor, TensorCore, TensorLinAlg, TensorStorage};

    #[test]
    fn test_tensor_multiplication_basic() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU).unwrap();

        let result = a.matmul(&b).unwrap();

        assert_eq!(*result.get(&[0, 0]).unwrap(), 19.0);  // 1*5 + 2*7 = 19
        assert_eq!(*result.get(&[0, 1]).unwrap(), 22.0);  // 1*6 + 2*8 = 22
        assert_eq!(*result.get(&[1, 0]).unwrap(), 43.0);  // 3*5 + 4*7 = 43
        assert_eq!(*result.get(&[1, 1]).unwrap(), 50.0);  // 3*6 + 4*8 = 50
    }

    #[test]
    fn test_tensor_multiplication_rectangular() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2], Device::CPU).unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 58.0);   // 1*7 + 2*9 + 3*11 = 58
        assert_eq!(*result.get(&[0, 1]).unwrap(), 64.0);   // 1*8 + 2*10 + 3*12 = 64
        assert_eq!(*result.get(&[1, 0]).unwrap(), 139.0);  // 4*7 + 5*9 + 6*11 = 139
        assert_eq!(*result.get(&[1, 1]).unwrap(), 154.0);  // 4*8 + 5*10 + 6*12 = 154
    }

    #[test]
    fn test_error_handling_tensor_multiplication() {
        let a = Tensor::<f32>::zeros(vec![2, 3], Device::CPU).unwrap();
        let b = Tensor::<f32>::zeros(vec![2, 2], Device::CPU).unwrap();

        let result = a.matmul(&b);
        assert!(result.is_err());

        let a_3d = Tensor::<f32>::zeros(vec![2, 3, 4], Device::CPU).unwrap();
        let b_3d = Tensor::<f32>::zeros(vec![2, 3, 4], Device::CPU).unwrap();
        let result = a_3d.matmul(&b_3d);
        assert!(result.is_err());
    }
}