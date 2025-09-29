use super::{Tensor, TensorError};
use num_traits::{Num, Float};

pub trait TensorElementwise<T> {
    fn exp(&self) -> Tensor<T> where T: Float;
    fn log(&self) -> Tensor<T> where T: Float;
    fn sqrt(&self) -> Tensor<T> where T: Float;
    fn square(&self) -> Tensor<T>;
    fn clip_max(&self, threshold: T) -> Tensor<T>;
    fn clip_min(&self, threshold: T) -> Tensor<T>;
    fn elementwise_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn elementwise_div(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn add_inplace(&mut self, other: &Tensor<T>) -> Result<(), TensorError>;
    fn sub(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn sub_inplace(&mut self, other: &Tensor<T>) -> Result<(), TensorError>;

    // Comparison operations that return binary masks
    fn greater_than(&self, threshold: T) -> Tensor<T>;
    fn greater_equal(&self, threshold: T) -> Tensor<T>;
    fn less_than(&self, threshold: T) -> Tensor<T>;
    fn less_equal(&self, threshold: T) -> Tensor<T>;
    fn equal(&self, threshold: T) -> Tensor<T>;
}

impl<T> TensorElementwise<T> for Tensor<T>
where
    T: Copy + Num + PartialOrd,
{
    /// Element-wise exponential of the tensor.
    /// 
    /// # Returns
    /// A new tensor with the exponential of each element
    fn exp(&self) -> Tensor<T>
    where
        T: Float,
    {
        let data: Vec<T> = self.data.iter().map(|&x| x.exp()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Element-wise natural logarithm of the tensor.
    /// 
    /// # Returns
    /// A new tensor with the natural logarithm of each element
    fn log(&self) -> Tensor<T>
    where
        T: Float,
    {
        let data: Vec<T> = self.data.iter().map(|&x| x.ln()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Element-wise square root of the tensor.
    /// 
    /// # Returns
    /// A new tensor with the square root of each element
    fn sqrt(&self) -> Tensor<T>
    where
        T: Float,
    {
        let data: Vec<T> = self.data.iter().map(|&x| x.sqrt()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Element-wise square of the tensor.
    /// 
    /// # Returns
    /// A new tensor with the square of each element
    fn square(&self) -> Tensor<T> {
        let data: Vec<T> = self.data.iter().map(|&x| x * x).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Clip elements of the tensor to a maximum value.
    /// 
    /// # Arguments
    /// * `threshold` - The maximum value to clip to
    /// 
    /// # Returns
    /// A new tensor with elements clipped to the maximum value
    fn clip_max(&self, threshold: T) -> Tensor<T> {
        let data: Vec<T> = self.data.iter().map(|&x| if x > threshold { threshold } else { x }).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Clip elements of the tensor to a minimum value.
    /// 
    /// # Arguments
    /// * `threshold` - The minimum value to clip to
    /// 
    /// # Returns
    /// A new tensor with elements clipped to the minimum value
    fn clip_min(&self, threshold: T) -> Tensor<T> {
        let data: Vec<T> = self.data.iter().map(|&x| if x < threshold { threshold } else { x }).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
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
    fn add_inplace(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        self.elementwise_op_inplace(other, |a, b| a + b)
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
    fn sub_inplace(&mut self, other: &Tensor<T>) -> Result<(), TensorError> {
        self.elementwise_op_inplace(other, |a, b| a - b)
    }

    /// Element-wise greater than comparison.
    ///
    /// # Arguments
    /// * `threshold` - Value to compare against
    ///
    /// # Returns
    /// A tensor with 1 where element > threshold, 0 otherwise
    fn greater_than(&self, threshold: T) -> Tensor<T> {
        let zero = T::zero();
        let one = T::one();
        let data: Vec<T> = self.data.iter()
            .map(|&x| if x > threshold { one } else { zero })
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Element-wise greater than or equal comparison.
    ///
    /// # Arguments
    /// * `threshold` - Value to compare against
    ///
    /// # Returns
    /// A tensor with 1 where element >= threshold, 0 otherwise
    fn greater_equal(&self, threshold: T) -> Tensor<T> {
        let zero = T::zero();
        let one = T::one();
        let data: Vec<T> = self.data.iter()
            .map(|&x| if x >= threshold { one } else { zero })
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Element-wise less than comparison.
    ///
    /// # Arguments
    /// * `threshold` - Value to compare against
    ///
    /// # Returns
    /// A tensor with 1 where element < threshold, 0 otherwise
    fn less_than(&self, threshold: T) -> Tensor<T> {
        let zero = T::zero();
        let one = T::one();
        let data: Vec<T> = self.data.iter()
            .map(|&x| if x < threshold { one } else { zero })
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Element-wise less than or equal comparison.
    ///
    /// # Arguments
    /// * `threshold` - Value to compare against
    ///
    /// # Returns
    /// A tensor with 1 where element <= threshold, 0 otherwise
    fn less_equal(&self, threshold: T) -> Tensor<T> {
        let zero = T::zero();
        let one = T::one();
        let data: Vec<T> = self.data.iter()
            .map(|&x| if x <= threshold { one } else { zero })
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Element-wise equality comparison.
    ///
    /// # Arguments
    /// * `threshold` - Value to compare against
    ///
    /// # Returns
    /// A tensor with 1 where element == threshold, 0 otherwise
    fn equal(&self, threshold: T) -> Tensor<T>
    where
        T: PartialEq,
    {
        let zero = T::zero();
        let one = T::one();
        let data: Vec<T> = self.data.iter()
            .map(|&x| if x == threshold { one } else { zero })
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_multiplication() {
        let a = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::<f32>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();

        let result = a.elementwise_mul(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0);  // 1 * 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 6.0);  // 2 * 3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 12.0); // 3 * 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 20.0); // 4 * 5
    }

    #[test]
    fn test_elementwise_division() {
        let a = Tensor::<f32>::from_data(vec![8.0, 12.0, 16.0, 20.0], vec![2, 2]).unwrap();
        let b = Tensor::<f32>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();

        let result = a.elementwise_div(&b).unwrap();
        assert_eq!(*result.get(&[0, 0]).unwrap(), 4.0);  // 8 / 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);  // 12 / 3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 4.0);  // 16 / 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0);  // 20 / 5
    }

    #[test]
    fn test_square() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = tensor.square();

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0);  // 1^2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);  // 2^2
        assert_eq!(*result.get(&[1, 0]).unwrap(), 9.0);  // 3^2
        assert_eq!(*result.get(&[1, 1]).unwrap(), 16.0); // 4^2
    }

    #[test]
    fn test_clip_max() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 5.0, 3.0, 8.0], vec![2, 2]).unwrap();
        let result = tensor.clip_max(4.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0); // 1 < 4, unchanged
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0); // 5 > 4, clipped to 4
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0); // 3 < 4, unchanged
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0); // 8 > 4, clipped to 4
    }

    #[test]
    fn test_clip_min() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 5.0, 3.0, 8.0], vec![2, 2]).unwrap();
        let result = tensor.clip_min(4.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 4.0); // 1 < 4, clipped to 4
        assert_eq!(*result.get(&[0, 1]).unwrap(), 5.0); // 5 > 4, unchanged
        assert_eq!(*result.get(&[1, 0]).unwrap(), 4.0); // 3 < 4, clipped to 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 8.0); // 8 > 4, unchanged
    }
}