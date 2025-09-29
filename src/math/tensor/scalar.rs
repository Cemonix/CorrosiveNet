use super::Tensor;
use std::ops::{Add, Sub, Mul, Div};

pub trait TensorScalar<T> {
    fn scalar_add(&self, scalar: T) -> Tensor<T>;
    fn scalar_add_inplace(&mut self, scalar: T);
    fn scalar_sub(&self, scalar: T) -> Tensor<T>;
    fn scalar_sub_inplace(&mut self, scalar: T);
    fn scalar_mul(&self, scalar: T) -> Tensor<T>;
    fn scalar_mul_inplace(&mut self, scalar: T);
    fn scalar_div(&self, scalar: T) -> Tensor<T>;
    fn scalar_div_inplace(&mut self, scalar: T);
}

impl<T> TensorScalar<T> for Tensor<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    /// Add a scalar value to all elements of the tensor.
    fn scalar_add(&self, scalar: T) -> Tensor<T> {
        self.scalar_op(scalar, |x, s| x + s)
    }

    /// Add a scalar value to all elements of the tensor in-place.
    fn scalar_add_inplace(&mut self, scalar: T) {
        for x in self.data.iter_mut() {
            *x = *x + scalar;
        }
    }

    /// Subtract a scalar value from all elements of the tensor.
    fn scalar_sub(&self, scalar: T) -> Tensor<T> {
        self.scalar_op(scalar, |x, s| x - s)
    }

    /// Subtract a scalar value from all elements of the tensor in-place.
    fn scalar_sub_inplace(&mut self, scalar: T) {
        for x in self.data.iter_mut() {
            *x = *x - scalar;
        }
    }

    /// Multiply all elements of the tensor by a scalar value.
    fn scalar_mul(&self, scalar: T) -> Tensor<T> {
        self.scalar_op(scalar, |x, s| x * s)
    }

    /// Multiply all elements of the tensor by a scalar value in-place.
    fn scalar_mul_inplace(&mut self, scalar: T) {
        for x in self.data.iter_mut() {
            *x = *x * scalar;
        }
    }

    /// Divide all elements of the tensor by a scalar value.
    fn scalar_div(&self, scalar: T) -> Tensor<T> {
        self.scalar_op(scalar, |x, s| x / s)
    }

    /// Divide all elements of the tensor by a scalar value in-place.
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
        let tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // Scalar addition
        let result = tensor.scalar_add(10.0);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 11.0);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 14.0);

        // Scalar multiplication
        let result = tensor.scalar_mul(2.0);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 8.0);

        // Scalar division
        let result = tensor.scalar_div(2.0);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 0.5);
        assert_eq!(*result.get(&[1, 1]).unwrap(), 2.0);
    }

    #[test]
    fn test_scalar_operations_inplace() {
        let mut tensor = Tensor::<f32>::from_data(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();

        tensor.scalar_mul_inplace(0.5);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 4.0);

        tensor.scalar_add_inplace(1.0);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 5.0);
    }
}