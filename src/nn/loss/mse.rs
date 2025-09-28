use std::ops::{Add, Div};

use crate::{math::{Matrix, MatrixError, MatrixElementwise, MatrixStats, MatrixScalar}, nn::LossFunction};
use num_traits::{Num, Float, NumCast};

pub struct MeanSquaredError;

impl MeanSquaredError {
    pub fn new() -> Self {
        MeanSquaredError
    }
}

impl<T> LossFunction<T> for MeanSquaredError
where
    T: Clone + Copy + Num + PartialOrd + NumCast + Float + Default + Add<Output = T> + Div<Output = T>,
{
    fn forward(&self, predictions: &Matrix<T>, targets: &Matrix<T>) -> Result<T, MatrixError> {
        // MSE = mean((predictions - targets)^2)

        // Compute (predictions - targets)
        let diff = predictions.sub(targets)?;

        // Compute (predictions - targets)^2
        let squared_diff = diff.square();

        // Compute mean of squared differences
        let sum = squared_diff.sum()?;
        let size = predictions.size();
        let n = NumCast::from(size).ok_or_else(|| MatrixError::new("Failed to convert matrix size to numeric type"))?;
        Ok(sum / n)
    }

    fn backward(&self, predictions: &Matrix<T>, targets: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        // MSE gradient: 2 * (predictions - targets) / n

        // Compute (predictions - targets)
        let diff = predictions.sub(targets)?;

        // Compute 2 * (predictions - targets)
        let two = NumCast::from(2u8).ok_or_else(|| MatrixError::new("Failed to convert 2 to numeric type"))?;
        let two_diff = diff.scalar_mul(two);

        // Compute 2 * (predictions - targets) / n
        let size = predictions.size();
        let n = NumCast::from(size).ok_or_else(|| MatrixError::new("Failed to convert matrix size to numeric type"))?;
        Ok(two_diff.scalar_div(n))
    }
}