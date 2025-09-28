use super::{Matrix, MatrixError};
use std::ops::{Add, Div};

pub trait MatrixStats<T> {
    fn sum(&self) -> Result<T, MatrixError>;
    fn mean(&self) -> Result<T, MatrixError> where T: From<usize>;
    fn max(&self) -> Result<T, MatrixError>;
    fn min(&self) -> Result<T, MatrixError>;
}

impl<T> MatrixStats<T> for Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Div<Output = T> + PartialOrd,
{
    /// Calculate the sum of all elements in the matrix.
    /// 
    /// # Returns
    /// The sum of all elements as type T.
    /// 
    /// # Errors
    /// Returns MatrixError if the matrix is empty.
    fn sum(&self) -> Result<T, MatrixError> {
        let mut total = T::default();
        for item in &self.data {
            total = total + *item;
        }
        Ok(total)
    }

    /// Calculate the mean (average) of all elements in the matrix.
    ///
    /// # Returns
    /// The mean of all elements as type T.
    ///
    /// # Errors
    /// Returns MatrixError if the matrix is empty.
    fn mean(&self) -> Result<T, MatrixError>
    where
        T: From<usize>,
    {
        if self.data.is_empty() {
            return Err(MatrixError::new("Cannot compute mean of empty matrix"));
        }

        let sum = self.sum()?;
        let count = self.size();
        Ok(sum / T::from(count))
    }

    /// Calculate the minimum value in the matrix.
    fn min(&self) -> Result<T, MatrixError> {
        if self.data.is_empty() {
            return Err(MatrixError::new("Cannot compute min of empty matrix"));
        }

        let mut min = self.data[0];
        for &item in &self.data[1..] {
            if item < min {
                min = item;
            }
        }
        Ok(min)
    }

    /// Calculate the maximum value in the matrix.
    fn max(&self) -> Result<T, MatrixError> {
        if self.data.is_empty() {
            return Err(MatrixError::new("Cannot compute max of empty matrix"));
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
        let matrix = Matrix::<usize>::from_data(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let sum = matrix.sum().unwrap();
        assert_eq!(sum, 10);
    }

    #[test]
    fn test_mean() {
        let matrix = Matrix::<usize>::from_data(vec![2, 4, 6, 8], vec![2, 2]).unwrap();
        let mean = matrix.mean().unwrap();
        assert_eq!(mean, 5); // (2+4+6+8)/4 = 20/4 = 5
    }

    #[test]
    fn test_min() {
        let matrix = Matrix::<usize>::from_data(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let min = matrix.min().unwrap();
        assert_eq!(min, 1);
    }

    #[test]
    fn test_max() {
        let matrix = Matrix::<usize>::from_data(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let max = matrix.max().unwrap();
        assert_eq!(max, 4);
    }

    #[test]
    fn test_stats_single_element() {
        let single_element = Matrix::<usize>::from_data(vec![42], vec![1]).unwrap();
        assert_eq!(single_element.sum().unwrap(), 42);
        assert_eq!(single_element.mean().unwrap(), 42);
        assert_eq!(single_element.min().unwrap(), 42);
        assert_eq!(single_element.max().unwrap(), 42);
    }
}