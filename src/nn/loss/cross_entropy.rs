use crate::{math::{Matrix, MatrixError, MatrixElementwise, MatrixStats, MatrixScalar}, nn::LossFunction};
use num_traits::{Num, Float, NumCast};
use std::ops::{Add, Div};

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }
}

impl<T> LossFunction<T> for CrossEntropyLoss
where
    T: Clone + Copy + Num + PartialOrd + NumCast + Float + Default + Add<Output = T> + Div<Output = T>,
{
    fn forward(&self, predictions: &Matrix<T>, targets: &Matrix<T>) -> Result<T, MatrixError> {
        // Cross-entropy: -mean(targets * log(predictions))

        // Compute log(predictions)
        let log_predictions = predictions.log();

        // Compute targets * log(predictions)
        let targets_log_pred = targets.elementwise_mul(&log_predictions)?;

        // Compute sum(targets * log(predictions))
        let sum = targets_log_pred.sum()?;

        // Compute -mean(targets * log(predictions)) = -sum / n
        let size = predictions.size();
        let n = NumCast::from(size).ok_or_else(|| MatrixError::new("Failed to convert matrix size to numeric type"))?;
        let mean = sum / n;
        Ok(-mean)
    }

    fn backward(&self, predictions: &Matrix<T>, targets: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        // Cross-entropy gradient: (predictions - targets) / n (for softmax + cross-entropy)

        // Compute (predictions - targets)
        let diff = predictions.sub(targets)?;

        // Compute (predictions - targets) / n
        let size = predictions.size();
        let n = NumCast::from(size).ok_or_else(|| MatrixError::new("Failed to convert matrix size to numeric type"))?;
        Ok(diff.scalar_div(n))
    }
}