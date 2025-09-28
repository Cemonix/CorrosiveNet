use super::{Matrix, MatrixError};
use std::ops::{Add, Sub, Mul, Div};

pub trait MatrixElementwise<T> {
    fn exp(&self) -> Matrix<T>;
    fn log(&self) -> Matrix<T>;
    fn sqrt(&self) -> Matrix<T>;
    fn square(&self) -> Matrix<T>;
    fn clip_max(&self, threshold: T) -> Matrix<T>;
    fn clip_min(&self, threshold: T) -> Matrix<T>;
    fn elementwise_mul(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
    fn elementwise_div(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
    fn add(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
    fn add_inplace(&mut self, other: &Matrix<T>) -> Result<(), MatrixError>;
    fn sub(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
    fn sub_inplace(&mut self, other: &Matrix<T>) -> Result<(), MatrixError>;
}

impl<T> MatrixElementwise<T> for Matrix<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + PartialOrd,
{
    fn exp(&self) -> Matrix<T> {
        todo!("Math functions like exp need to be implemented")
    }

    fn log(&self) -> Matrix<T> {
        todo!("Math functions like log need to be implemented")
    }

    fn sqrt(&self) -> Matrix<T> {
        todo!("Math functions like sqrt need to be implemented")
    }

    fn square(&self) -> Matrix<T> {
        let data: Vec<T> = self.data.iter().map(|&x| x * x).collect();
        Matrix {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    fn clip_max(&self, threshold: T) -> Matrix<T> {
        let data: Vec<T> = self.data.iter().map(|&x| if x > threshold { threshold } else { x }).collect();
        Matrix {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    fn clip_min(&self, threshold: T) -> Matrix<T> {
        let data: Vec<T> = self.data.iter().map(|&x| if x < threshold { threshold } else { x }).collect();
        Matrix {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Element-wise multiplication of two matrices.
    ///
    /// # Arguments
    /// * `other` - The matrix to multiply element-wise with this one
    ///
    /// # Returns
    /// A new matrix containing the element-wise product
    ///
    /// # Errors
    /// When shapes do not match
    fn elementwise_mul(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        self.elementwise_op(other, |a, b| a * b)
    }

    /// Element-wise division of two matrices.
    ///
    /// # Arguments
    /// * `other` - The matrix to divide element-wise with this one
    ///
    /// # Returns
    /// A new matrix containing the element-wise quotient
    ///
    /// # Errors
    /// When shapes do not match
    fn elementwise_div(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        self.elementwise_op(other, |a, b| a / b)
    }

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_multiplication() {
        let a = Matrix::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Matrix::<f32>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();

        let result = a.elementwise_mul(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0);  // 1 * 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 6.0);  // 2 * 3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 12.0); // 3 * 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 20.0); // 4 * 5
    }

    #[test]
    fn test_elementwise_division() {
        let a = Matrix::<f32>::from_data(vec![8.0, 12.0, 16.0, 20.0], vec![2, 2]).unwrap();
        let b = Matrix::<f32>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();

        let result = a.elementwise_div(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 4.0);  // 8 / 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);  // 12 / 3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 4.0);  // 16 / 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0);  // 20 / 5
    }

    #[test]
    fn test_square() {
        let matrix = Matrix::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = matrix.square();

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0);  // 1^2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);  // 2^2
        assert_eq!(*result.get(&[1, 0]).unwrap(), 9.0);  // 3^2
        assert_eq!(*result.get(&[1, 1]).unwrap(), 16.0); // 4^2
    }

    #[test]
    fn test_clip_max() {
        let matrix = Matrix::<f32>::from_data(vec![1.0, 5.0, 3.0, 8.0], vec![2, 2]).unwrap();
        let result = matrix.clip_max(4.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0); // 1 < 4, unchanged
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0); // 5 > 4, clipped to 4
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0); // 3 < 4, unchanged
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0); // 8 > 4, clipped to 4
    }

    #[test]
    fn test_clip_min() {
        let matrix = Matrix::<f32>::from_data(vec![1.0, 5.0, 3.0, 8.0], vec![2, 2]).unwrap();
        let result = matrix.clip_min(4.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 4.0); // 1 < 4, clipped to 4
        assert_eq!(*result.get(&[0, 1]).unwrap(), 5.0); // 5 > 4, unchanged
        assert_eq!(*result.get(&[1, 0]).unwrap(), 4.0); // 3 < 4, clipped to 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 8.0); // 8 > 4, unchanged
    }
}