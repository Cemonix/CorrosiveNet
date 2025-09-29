use super::{Tensor, TensorError};
use std::ops::{Add, Div};

pub trait TensorStats<T> {
    fn sum(&self) -> Result<T, TensorError>;
    fn mean(&self) -> Result<T, TensorError> where T: From<usize>;
    fn max(&self) -> Result<T, TensorError>;
    fn min(&self) -> Result<T, TensorError>;
}

impl<T> TensorStats<T> for Tensor<T>
where
    T: Copy + Default + Add<Output = T> + Div<Output = T> + PartialOrd,
{
    /// Calculate the sum of all elements in the tensor.
    /// 
    /// # Returns
    /// The sum of all elements as type T.
    /// 
    /// # Errors
    /// Returns TensorError if the tensor is empty.
    fn sum(&self) -> Result<T, TensorError> {
        let mut total = T::default();
        for item in &self.data {
            total = total + *item;
        }
        Ok(total)
    }

    /// Calculate the mean (average) of all elements in the tensor.
    ///
    /// # Returns
    /// The mean of all elements as type T.
    ///
    /// # Errors
    /// Returns TensorError if the tensor is empty.
    fn mean(&self) -> Result<T, TensorError>
    where
        T: From<usize>,
    {
        if self.data.is_empty() {
            return Err(TensorError::new("Cannot compute mean of empty tensor"));
        }

        let sum = self.sum()?;
        let count = self.size();
        Ok(sum / T::from(count))
    }

    /// Calculate the minimum value in the tensor.
    fn min(&self) -> Result<T, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError::new("Cannot compute min of empty tensor"));
        }

        let mut min = self.data[0];
        for &item in &self.data[1..] {
            if item < min {
                min = item;
            }
        }
        Ok(min)
    }

    /// Calculate the maximum value in the tensor.
    fn max(&self) -> Result<T, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError::new("Cannot compute max of empty tensor"));
        }

        let mut max = self.data[0];
        for &item in &self.data[1..] {
            if item > max {
                max = item;
            }
        }
        Ok(max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let tensor = Tensor::<usize>::from_data(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let sum = tensor.sum().unwrap();
        assert_eq!(sum, 10);
    }

    #[test]
    fn test_mean() {
        let tensor = Tensor::<usize>::from_data(vec![2, 4, 6, 8], vec![2, 2]).unwrap();
        let mean = tensor.mean().unwrap();
        assert_eq!(mean, 5); // (2+4+6+8)/4 = 20/4 = 5
    }

    #[test]
    fn test_min() {
        let tensor = Tensor::<usize>::from_data(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let min = tensor.min().unwrap();
        assert_eq!(min, 1);
    }

    #[test]
    fn test_max() {
        let tensor = Tensor::<usize>::from_data(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let max = tensor.max().unwrap();
        assert_eq!(max, 4);
    }

    #[test]
    fn test_stats_single_element() {
        let single_element = Tensor::<usize>::from_data(vec![42], vec![1]).unwrap();
        assert_eq!(single_element.sum().unwrap(), 42);
        assert_eq!(single_element.mean().unwrap(), 42);
        assert_eq!(single_element.min().unwrap(), 42);
        assert_eq!(single_element.max().unwrap(), 42);
    }
}