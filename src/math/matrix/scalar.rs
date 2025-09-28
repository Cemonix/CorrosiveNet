use super::Matrix;
use std::ops::{Add, Sub, Mul, Div};

pub trait MatrixScalar<T> {
    fn scalar_add(&self, scalar: T) -> Matrix<T>;
    fn scalar_add_inplace(&mut self, scalar: T);
    fn scalar_sub(&self, scalar: T) -> Matrix<T>;
    fn scalar_sub_inplace(&mut self, scalar: T);
    fn scalar_mul(&self, scalar: T) -> Matrix<T>;
    fn scalar_mul_inplace(&mut self, scalar: T);
    fn scalar_div(&self, scalar: T) -> Matrix<T>;
    fn scalar_div_inplace(&mut self, scalar: T);
}

impl<T> MatrixScalar<T> for Matrix<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    /// Add a scalar value to all elements of the matrix.
    fn scalar_add(&self, scalar: T) -> Matrix<T> {
        self.scalar_op(scalar, |x, s| x + s)
    }

    /// Add a scalar value to all elements of the matrix in-place.
    fn scalar_add_inplace(&mut self, scalar: T) {
        for x in self.data.iter_mut() {
            *x = *x + scalar;
        }
    }

    /// Subtract a scalar value from all elements of the matrix.
    fn scalar_sub(&self, scalar: T) -> Matrix<T> {
        self.scalar_op(scalar, |x, s| x - s)
    }

    /// Subtract a scalar value from all elements of the matrix in-place.
    fn scalar_sub_inplace(&mut self, scalar: T) {
        for x in self.data.iter_mut() {
            *x = *x - scalar;
        }
    }

    /// Multiply all elements of the matrix by a scalar value.
    fn scalar_mul(&self, scalar: T) -> Matrix<T> {
        self.scalar_op(scalar, |x, s| x * s)
    }

    /// Multiply all elements of the matrix by a scalar value in-place.
    fn scalar_mul_inplace(&mut self, scalar: T) {
        for x in self.data.iter_mut() {
            *x = *x * scalar;
        }
    }

    /// Divide all elements of the matrix by a scalar value.
    fn scalar_div(&self, scalar: T) -> Matrix<T> {
        self.scalar_op(scalar, |x, s| x / s)
    }

    /// Divide all elements of the matrix by a scalar value in-place.
    fn scalar_div_inplace(&mut self, scalar: T) {
        for x in self.data.iter_mut() {
            *x = *x / scalar;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_operations() {
        let matrix = Matrix::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // Scalar addition
        let result = matrix.scalar_add(10.0);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 14.0);

        // Scalar multiplication
        let result = matrix.scalar_mul(2.0);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 8.0);

        // Scalar division
        let result = matrix.scalar_div(2.0);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 0.5);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 2.0);
    }

    #[test]
    fn test_scalar_operations_inplace() {
        let mut matrix = Matrix::<f32>::from_data(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();

        matrix.scalar_mul_inplace(0.5);
        assert_eq!(*matrix.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*matrix.get(&[1, 1]).unwrap(), 4.0);

        matrix.scalar_add_inplace(1.0);
        assert_eq!(*matrix.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*matrix.get(&[1, 1]).unwrap(), 5.0);
    }
}