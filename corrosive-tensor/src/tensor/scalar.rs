use super::{Tensor, TensorNum};

pub trait TensorScalar<T> {
    fn scalar_add(&self, scalar: T) -> Tensor<T>;
    fn scalar_add_mut(&mut self, scalar: T);
    fn scalar_sub(&self, scalar: T) -> Tensor<T>;
    fn scalar_sub_mut(&mut self, scalar: T);
    fn scalar_mul(&self, scalar: T) -> Tensor<T>;
    fn scalar_mul_mut(&mut self, scalar: T);
    fn scalar_div(&self, scalar: T) -> Tensor<T>;
    fn scalar_div_mut(&mut self, scalar: T);
}

impl<T> TensorScalar<T> for Tensor<T>
where
    T: TensorNum,
{
    /// Add a scalar value to all elements of the tensor.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to add to each element
    ///
    /// # Returns
    /// A new tensor with the scalar added to all elements
    fn scalar_add(&self, scalar: T) -> Tensor<T> {
        self.scalar_op(scalar, |x, s| x + s)
    }

    /// Add a scalar value to all elements of the tensor in-place.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to add to each element
    fn scalar_add_mut(&mut self, scalar: T) {
        self.data.iter_mut().for_each(|x| *x = *x + scalar);
    }

    /// Subtract a scalar value from all elements of the tensor.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to subtract from each element
    ///
    /// # Returns
    /// A new tensor with the scalar subtracted from all elements
    fn scalar_sub(&self, scalar: T) -> Tensor<T> {
        self.scalar_op(scalar, |x, s| x - s)
    }

    /// Subtract a scalar value from all elements of the tensor in-place.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to subtract from each element
    fn scalar_sub_mut(&mut self, scalar: T) {
        self.data.iter_mut().for_each(|x| *x = *x - scalar);
    }

    /// Multiply all elements of the tensor by a scalar value.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to multiply each element by
    ///
    /// # Returns
    /// A new tensor with all elements multiplied by the scalar
    fn scalar_mul(&self, scalar: T) -> Tensor<T> {
        self.scalar_op(scalar, |x, s| x * s)
    }

    /// Multiply all elements of the tensor by a scalar value in-place.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to multiply each element by
    fn scalar_mul_mut(&mut self, scalar: T) {
        self.data.iter_mut().for_each(|x| *x = *x * scalar);
    }

    /// Divide all elements of the tensor by a scalar value.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to divide each element by
    ///
    /// # Returns
    /// A new tensor with all elements divided by the scalar
    fn scalar_div(&self, scalar: T) -> Tensor<T> {
        self.scalar_op(scalar, |x, s| x / s)
    }

    /// Divide all elements of the tensor by a scalar value in-place.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to divide each element by
    fn scalar_div_mut(&mut self, scalar: T) {
        self.data.iter_mut().for_each(|x| *x = *x / scalar);
    }
}

impl<T> Tensor<T> {
    /// Generic helper for scalar operations.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to operate with
    /// * `op` - The binary operation to apply element-wise
    ///
    /// # Returns
    /// A new tensor containing the result of the scalar operation
    fn scalar_op<F>(&self, scalar: T, op: F) -> Tensor<T>
    where
        F: Fn(T, T) -> T,
        T: Copy,
    {
        let data: Vec<T> = self.data.iter().map(|&x| op(x, scalar)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: Self::calculate_strides(&self.shape),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Tensor, TensorCore, TensorStorage, TensorScalar};

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
    fn test_scalar_operations_mut() {
        let mut tensor = Tensor::<f32>::from_data(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]).unwrap();

        tensor.scalar_mul_mut(0.5);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 4.0);

        tensor.scalar_add_mut(1.0);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 5.0);
    }
}