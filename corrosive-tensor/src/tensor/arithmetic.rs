use super::{Tensor, TensorError, TensorNum};

pub trait TensorArithmetic<T> {
    fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn add_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError>;
    fn sub(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn sub_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError>;
    fn elementwise_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn elementwise_mul_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError>;
    fn elementwise_div(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn elementwise_div_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError>;
}

impl<T> TensorArithmetic<T> for Tensor<T>
where
    T: TensorNum,
{
    /// Element-wise addition of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to add to this one
    ///
    /// # Returns
    /// A new tensor containing the element-wise sum
    ///
    /// # Errors
    /// When shapes do not match
    fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        self.elementwise_op(other, |a, b| a + b)
    }

    /// In-place element-wise addition of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to add to this one
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match
    fn add_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        self.elementwise_op_mut(other, |a, b| a + b)
    }

    /// Element-wise subtraction of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from this one
    ///
    /// # Returns
    /// A new tensor containing the element-wise difference
    ///
    /// # Errors
    /// When shapes do not match
    fn sub(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        self.elementwise_op(other, |a, b| a - b)
    }

    /// In-place element-wise subtraction of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to subtract from this one
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match
    fn sub_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        self.elementwise_op_mut(other, |a, b| a - b)
    }

    /// Element-wise multiplication of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply element-wise with this one
    ///
    /// # Returns
    /// A new tensor containing the element-wise product
    ///
    /// # Errors
    /// When shapes do not match
    fn elementwise_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        self.elementwise_op(other, |a, b| a * b)
    }

    /// In-place element-wise multiplication of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply element-wise with this one
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match
    fn elementwise_mul_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        self.elementwise_op_mut(other, |a, b| a * b)
    }

    /// Element-wise division of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to divide element-wise with this one
    ///
    /// # Returns
    /// A new tensor containing the element-wise quotient
    ///
    /// # Errors
    /// When shapes do not match
    fn elementwise_div(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        self.elementwise_op(other, |a, b| a / b)
    }

    /// In-place element-wise division of two tensors.
    ///
    /// # Arguments
    /// * `other` - The tensor to divide element-wise with this one
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match
    fn elementwise_div_mut(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        self.elementwise_op_mut(other, |a, b| a / b)
    }
}

impl<T> Tensor<T> {
    /// Generic helper for element-wise operations between two tensors.
    ///
    /// # Arguments
    /// * `other` - The other tensor to operate with
    /// * `op` - The binary operation to apply element-wise
    ///
    /// # Returns
    /// A new tensor containing the result of the element-wise operation
    ///
    /// # Errors
    /// When shapes do not match
    pub(super) fn elementwise_op<F>(&self, other: &Tensor<T>, op: F) -> Result<Tensor<T>, TensorError>
    where
        F: Fn(T, T) -> T,
        T: Copy,
    {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for operation"));
        }

        if !self.has_same_device(other) {
            return Err(TensorError::new("Tensors are on different devices"));
        }

        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| op(*a, *b))
            .collect();

        Ok(Tensor {
            data,
            shape: self.shape.clone(),
            strides: Self::calculate_strides(&self.shape),
            device: self.device.clone(),
        })
    }

    /// Generic helper for in-place element-wise operations between two tensors.
    ///
    /// # Arguments
    /// * `other` - The other tensor to operate with
    /// * `op` - The binary operation to apply element-wise
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match
    pub(super) fn elementwise_op_mut<F>(&mut self, other: &Tensor<T>, op: F) -> Result<(), TensorError>
    where
        F: Fn(T, T) -> T,
        T: Copy,
    {
        if self.shape != other.shape {
            return Err(TensorError::new("Shapes do not match for operation"));
        }

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = op(*a, *b);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Device, Tensor, TensorArithmetic, TensorCore, TensorStorage};

    #[test]
    fn test_element_wise_addition() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU).unwrap();

        let result = a.add(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 6.0);  // 1 + 5
        assert_eq!(*result.get(&[0, 1]).unwrap(), 8.0);  // 2 + 6
        assert_eq!(*result.get(&[1, 0]).unwrap(), 10.0); // 3 + 7
        assert_eq!(*result.get(&[1, 1]).unwrap(), 12.0); // 4 + 8
    }

    #[test]
    fn test_element_wise_addition_mut() {
        let mut a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], Device::CPU).unwrap();

        a.add_mut(&b).unwrap();
        assert_eq!(*a.get(&[0, 0]).unwrap(), 6.0);
        assert_eq!(*a.get(&[1, 1]).unwrap(), 12.0);
    }

    #[test]
    fn test_element_wise_subtraction() {
        let a = Tensor::<f32>::from_data(vec![10.0, 8.0, 6.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();

        let result = a.sub(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 9.0);  // 10 - 1
        assert_eq!(*result.get(&[0, 1]).unwrap(), 6.0);  // 8 - 2
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0);  // 6 - 3
        assert_eq!(*result.get(&[1, 1]).unwrap(), 0.0);  // 4 - 4
    }

    #[test]
    fn test_elementwise_multiplication() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], Device::CPU).unwrap();

        let result = a.elementwise_mul(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0);  // 1 * 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 6.0);  // 2 * 3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 12.0); // 3 * 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 20.0); // 4 * 5
    }

    #[test]
    fn test_elementwise_division() {
        let a = Tensor::<f32>::from_data(vec![8.0, 12.0, 16.0, 20.0], vec![2, 2], Device::CPU).unwrap();
        let b = Tensor::<f32>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], Device::CPU).unwrap();

        let result = a.elementwise_div(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 4.0);  // 8 / 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);  // 12 / 3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 4.0);  // 16 / 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0);  // 20 / 5
    }

    #[test]
    fn test_error_handling_shape_mismatch() {
        let a = Tensor::<f32>::zeros(vec![2, 3], Device::CPU).unwrap();
        let b = Tensor::<f32>::zeros(vec![3, 2], Device::CPU).unwrap();

        let result = a.add(&b);
        assert!(result.is_err());

        let mut a_mut = Tensor::<f32>::zeros(vec![2, 3], Device::CPU).unwrap();
        let result = a_mut.add_mut(&b);
        assert!(result.is_err());
    }
}