use super::{Tensor, TensorNum};

pub trait TensorComparison<T> {
    fn greater_than(&self, threshold: T) -> Tensor<T>;
    fn greater_equal(&self, threshold: T) -> Tensor<T>;
    fn less_than(&self, threshold: T) -> Tensor<T>;
    fn less_equal(&self, threshold: T) -> Tensor<T>;
    fn equal(&self, threshold: T) -> Tensor<T>;
    fn not_equal(&self, threshold: T) -> Tensor<T>;
    fn clip_max(&self, threshold: T) -> Tensor<T>;
    fn clip_min(&self, threshold: T) -> Tensor<T>;
    fn clip(&self, min_val: T, max_val: T) -> Tensor<T>;
}

impl<T> TensorComparison<T> for Tensor<T>
where
    T: TensorNum,
{
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
        self.comparison_op(threshold, |x, thresh| if x > thresh { one } else { zero })
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
        self.comparison_op(threshold, |x, thresh| if x >= thresh { one } else { zero })
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
        self.comparison_op(threshold, |x, thresh| if x < thresh { one } else { zero })
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
        self.comparison_op(threshold, |x, thresh| if x <= thresh { one } else { zero })
    }

    /// Element-wise equality comparison.
    ///
    /// # Arguments
    /// * `threshold` - Value to compare against
    ///
    /// # Returns
    /// A tensor with 1 where element == threshold, 0 otherwise
    fn equal(&self, threshold: T) -> Tensor<T> {
        let zero = T::zero();
        let one = T::one();
        self.comparison_op(threshold, |x, thresh| if x == thresh { one } else { zero })
    }

    /// Element-wise not equal comparison.
    ///
    /// # Arguments
    /// * `threshold` - Value to compare against
    ///
    /// # Returns
    /// A tensor with 1 where element != threshold, 0 otherwise
    fn not_equal(&self, threshold: T) -> Tensor<T> {
        let zero = T::zero();
        let one = T::one();
        self.comparison_op(threshold, |x, thresh| if x != thresh { one } else { zero })
    }

    /// Clip elements of the tensor to a maximum value.
    ///
    /// # Arguments
    /// * `threshold` - The maximum value to clip to
    ///
    /// # Returns
    /// A new tensor with elements clipped to the maximum value
    fn clip_max(&self, threshold: T) -> Tensor<T> {
        self.comparison_op(threshold, |x, thresh| if x > thresh { thresh } else { x })
    }

    /// Clip elements of the tensor to a minimum value.
    ///
    /// # Arguments
    /// * `threshold` - The minimum value to clip to
    ///
    /// # Returns
    /// A new tensor with elements clipped to the minimum value
    fn clip_min(&self, threshold: T) -> Tensor<T> {
        self.comparison_op(threshold, |x, thresh| if x < thresh { thresh } else { x })
    }

    /// Clip elements of the tensor between minimum and maximum values.
    ///
    /// # Arguments
    /// * `min_val` - The minimum value to clip to
    /// * `max_val` - The maximum value to clip to
    ///
    /// # Returns
    /// A new tensor with elements clipped between min and max values
    fn clip(&self, min_val: T, max_val: T) -> Tensor<T> {
        let data: Vec<T> = self.data.iter()
            .map(|&x| {
                if x < min_val {
                    min_val
                } else if x > max_val {
                    max_val
                } else {
                    x
                }
            })
            .collect();

        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            device: self.device.clone(),
        }
    }
}

impl<T> Tensor<T> {
    /// Generic helper for comparison operations with a scalar.
    ///
    /// # Arguments
    /// * `threshold` - The scalar value to compare against
    /// * `op` - The comparison operation to apply
    ///
    /// # Returns
    /// A new tensor with the comparison results
    fn comparison_op<F>(&self, threshold: T, op: F) -> Tensor<T>
    where
        F: Fn(T, T) -> T,
        T: Copy,
    {
        let data: Vec<T> = self.data.iter()
            .map(|&x| op(x, threshold))
            .collect();

        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            device: self.device.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Device, Tensor, TensorComparison, TensorCore, TensorStorage};


    #[test]
    fn test_greater_than() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 3.0, 5.0, 2.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.greater_than(3.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 0.0); // 1 > 3? No
        assert_eq!(*result.get(&[0, 1]).unwrap(), 0.0); // 3 > 3? No
        assert_eq!(*result.get(&[1, 0]).unwrap(), 1.0); // 5 > 3? Yes
        assert_eq!(*result.get(&[1, 1]).unwrap(), 0.0); // 2 > 3? No
    }

    #[test]
    fn test_greater_equal() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 3.0, 5.0, 2.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.greater_equal(3.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 0.0); // 1 >= 3? No
        assert_eq!(*result.get(&[0, 1]).unwrap(), 1.0); // 3 >= 3? Yes
        assert_eq!(*result.get(&[1, 0]).unwrap(), 1.0); // 5 >= 3? Yes
        assert_eq!(*result.get(&[1, 1]).unwrap(), 0.0); // 2 >= 3? No
    }

    #[test]
    fn test_less_than() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 3.0, 5.0, 2.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.less_than(3.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0); // 1 < 3? Yes
        assert_eq!(*result.get(&[0, 1]).unwrap(), 0.0); // 3 < 3? No
        assert_eq!(*result.get(&[1, 0]).unwrap(), 0.0); // 5 < 3? No
        assert_eq!(*result.get(&[1, 1]).unwrap(), 1.0); // 2 < 3? Yes
    }

    #[test]
    fn test_equal() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 3.0, 3.0, 2.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.equal(3.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 0.0); // 1 == 3? No
        assert_eq!(*result.get(&[0, 1]).unwrap(), 1.0); // 3 == 3? Yes
        assert_eq!(*result.get(&[1, 0]).unwrap(), 1.0); // 3 == 3? Yes
        assert_eq!(*result.get(&[1, 1]).unwrap(), 0.0); // 2 == 3? No
    }

    #[test]
    fn test_clip_max() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 5.0, 3.0, 8.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.clip_max(4.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0); // 1 < 4, unchanged
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0); // 5 > 4, clipped to 4
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0); // 3 < 4, unchanged
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0); // 8 > 4, clipped to 4
    }

    #[test]
    fn test_clip_min() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 5.0, 3.0, 8.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.clip_min(4.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 4.0); // 1 < 4, clipped to 4
        assert_eq!(*result.get(&[0, 1]).unwrap(), 5.0); // 5 > 4, unchanged
        assert_eq!(*result.get(&[1, 0]).unwrap(), 4.0); // 3 < 4, clipped to 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 8.0); // 8 > 4, unchanged
    }

    #[test]
    fn test_clip() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 5.0, 3.0, 8.0], vec![2, 2], Device::CPU).unwrap();
        let result = tensor.clip(2.0, 6.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0); // 1 < 2, clipped to 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 5.0); // 5 in [2,6], unchanged
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0); // 3 in [2,6], unchanged
        assert_eq!(*result.get(&[1, 1]).unwrap(), 6.0); // 8 > 6, clipped to 6
    }
}