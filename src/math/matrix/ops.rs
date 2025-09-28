use super::{Matrix, MatrixError};
use std::ops::{Add, Sub, AddAssign, Mul};

pub trait MatrixOps<T> {
    fn add(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
    fn add_inplace(&mut self, other: &Matrix<T>) -> Result<(), MatrixError>;
    fn sub(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
    fn sub_inplace(&mut self, other: &Matrix<T>) -> Result<(), MatrixError>;
    fn matmul(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
}

impl<T> MatrixOps<T> for Matrix<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + AddAssign + Mul<Output = T> + Default,
{
    /// Element-wise addition of two matrices.
    ///
    /// # Arguments
    /// * `other` - The matrix to add to this one
    ///
    /// # Returns
    /// A new matrix containing the element-wise sum
    ///
    /// # Errors
    /// When shapes do not match
    fn add(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        self.elementwise_op(other, |a, b| a + b)
    }

    /// In-place element-wise addition of two matrices.
    ///
    /// # Arguments
    /// * `other` - The matrix to add to this one
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match
    fn add_inplace(&mut self, other: &Matrix<T>) -> Result<(), MatrixError> {
        self.elementwise_op_inplace(other, |a, b| a + b)
    }

    /// Element-wise subtraction of two matrices.
    ///
    /// # Arguments
    /// * `other` - The matrix to subtract from this one
    ///
    /// # Returns
    /// A new matrix containing the element-wise difference
    ///
    /// # Errors
    /// When shapes do not match
    fn sub(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        self.elementwise_op(other, |a, b| a - b)
    }

    /// In-place element-wise subtraction of two matrices.
    ///
    /// # Arguments
    /// * `other` - The matrix to subtract from this one
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match
    fn sub_inplace(&mut self, other: &Matrix<T>) -> Result<(), MatrixError> {
        self.elementwise_op_inplace(other, |a, b| a - b)
    }

    /// Perform matrix multiplication between two 2D matrices.
    ///
    /// Computes the matrix product C = A × B where:
    /// - A is an m×k matrix (self)
    /// - B is a k×n matrix (other)
    /// - C is an m×n matrix (result)
    ///
    /// # Arguments
    /// * `other` - The right-hand matrix to multiply with
    ///
    /// # Returns
    /// A new matrix containing the matrix product
    ///
    /// # Errors
    /// Returns an error if:
    /// - Either matrix is not 2D
    /// - The inner dimensions don't match (A.cols ≠ B.rows)
    fn matmul(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.shape().len() != 2 || other.shape().len() != 2 {
            return Err(MatrixError::new(&format!(
                "Matrices must be 2D, got {}D and {}D", self.shape().len(), other.shape().len()
            )));
        }

        if self.shape()[1] != other.shape()[0] {
            return Err(MatrixError::new(&format!(
                "Matrix dimension mismatch: cannot multiply {}×{} with {}×{} (inner dimensions {} ≠ {})",
                self.shape()[0], self.shape()[1], other.shape()[0], other.shape()[1],
                self.shape()[1], other.shape()[0]
            )));
        }

        let rows = self.shape()[0];
        let cols = other.shape()[1];
        let inner = self.shape()[1];

        let result_shape = vec![rows, cols];
        let mut result = Matrix::zeros(result_shape)?;

        let self_data = &self.data;
        let other_data = &other.data;
        let result_strides = result.strides().to_vec();
        let result_data = &mut result.data;

        for i in 0..rows {
            for j in 0..cols {
                let mut sum = T::default();

                for k in 0..inner {
                    let self_idx = i * self.strides()[0] + k * self.strides()[1];
                    let other_idx = k * other.strides()[0] + j * other.strides()[1];
                    sum += self_data[self_idx] * other_data[other_idx];
                }

                let result_idx = i * result_strides[0] + j * result_strides[1];
                result_data[result_idx] = sum;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_wise_addition() {
        let a = Matrix::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Matrix::<f32>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        let result = a.add(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 6.0);  // 1 + 5
        assert_eq!(*result.get(&[0, 1]).unwrap(), 8.0);  // 2 + 6
        assert_eq!(*result.get(&[1, 0]).unwrap(), 10.0); // 3 + 7
        assert_eq!(*result.get(&[1, 1]).unwrap(), 12.0); // 4 + 8
    }

    #[test]
    fn test_element_wise_addition_inplace() {
        let mut a = Matrix::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Matrix::<f32>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        a.add_inplace(&b).unwrap();
        assert_eq!(*a.get(&[0, 0]).unwrap(), 6.0);
        assert_eq!(*a.get(&[1, 1]).unwrap(), 12.0);
    }

    #[test]
    fn test_element_wise_subtraction() {
        let a = Matrix::<f32>::from_data(vec![10.0, 8.0, 6.0, 4.0], vec![2, 2]).unwrap();
        let b = Matrix::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let result = a.sub(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 9.0);  // 10 - 1
        assert_eq!(*result.get(&[0, 1]).unwrap(), 6.0);  // 8 - 2
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0);  // 6 - 3
        assert_eq!(*result.get(&[1, 1]).unwrap(), 0.0);  // 4 - 4
    }

    #[test]
    fn test_matrix_multiplication_basic() {
        // Test basic 2x2 matrix multiplication
        let a = Matrix::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Matrix::<f32>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        let result = a.matmul(&b).unwrap();

        // Expected result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //                  [[19, 22], [43, 50]]
        assert_eq!(*result.get(&[0, 0]).unwrap(), 19.0);  // 1*5 + 2*7 = 19
        assert_eq!(*result.get(&[0, 1]).unwrap(), 22.0);  // 1*6 + 2*8 = 22
        assert_eq!(*result.get(&[1, 0]).unwrap(), 43.0);  // 3*5 + 4*7 = 43
        assert_eq!(*result.get(&[1, 1]).unwrap(), 50.0);  // 3*6 + 4*8 = 50
    }

    #[test]
    fn test_matrix_multiplication_rectangular() {
        // Test 2x3 * 3x2 = 2x2
        let a = Matrix::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Matrix::<f32>::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //          [[58, 64], [139, 154]]
        assert_eq!(*result.get(&[0, 0]).unwrap(), 58.0);   // 1*7 + 2*9 + 3*11 = 58
        assert_eq!(*result.get(&[0, 1]).unwrap(), 64.0);   // 1*8 + 2*10 + 3*12 = 64
        assert_eq!(*result.get(&[1, 0]).unwrap(), 139.0);  // 4*7 + 5*9 + 6*11 = 139
        assert_eq!(*result.get(&[1, 1]).unwrap(), 154.0);  // 4*8 + 5*10 + 6*12 = 154
    }

    #[test]
    fn test_matrix_multiplication_identity() {
        let matrix = Matrix::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let identity = Matrix::<f32>::from_data(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let result = matrix.matmul(&identity).unwrap();

        // Multiplying by identity should return the original matrix
        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*result.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_matrix_multiplication_zeros() {
        let matrix = Matrix::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let zeros = Matrix::<f32>::zeros(vec![2, 2]).unwrap();

        let result = matrix.matmul(&zeros).unwrap();

        // Multiplying by zeros should return all zeros
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(*result.get(&[i, j]).unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn test_error_handling_matrix_multiplication() {
        let a = Matrix::<f32>::zeros(vec![2, 3]).unwrap();
        let b = Matrix::<f32>::zeros(vec![2, 2]).unwrap(); // Wrong inner dimension

        let result = a.matmul(&b);
        assert!(result.is_err());

        // Test 3D matrix (not supported)
        let a_3d = Matrix::<f32>::zeros(vec![2, 3, 4]).unwrap();
        let b_3d = Matrix::<f32>::zeros(vec![2, 3, 4]).unwrap();
        let result = a_3d.matmul(&b_3d);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_handling_element_wise_operations() {
        let a = Matrix::<f32>::zeros(vec![2, 3]).unwrap();
        let b = Matrix::<f32>::zeros(vec![3, 2]).unwrap(); // Different shape

        // Test shape mismatch
        let result = a.add(&b);
        assert!(result.is_err());

        let mut a_mut = Matrix::<f32>::zeros(vec![2, 3]).unwrap();
        let result = a_mut.add_inplace(&b);
        assert!(result.is_err());
    }
}