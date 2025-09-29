use crate::{math::{Tensor, TensorError, TensorElementwise, TensorStats, TensorScalar}, nn::LossFunction};
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
    fn forward(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<T, TensorError> {
        // Cross-entropy: -mean(targets * log(predictions))

        // Compute log(predictions)
        let log_predictions = predictions.log();

        // Compute targets * log(predictions)
        let targets_log_pred = targets.elementwise_mul(&log_predictions)?;

        // Compute sum(targets * log(predictions))
        let sum = targets_log_pred.sum()?;

        // Compute -mean(targets * log(predictions)) = -sum / n
        let size = predictions.size();
        let n = NumCast::from(size).ok_or_else(|| TensorError::new("Failed to convert matrix size to numeric type"))?;
        let mean = sum / n;
        Ok(-mean)
    }

    fn backward(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        // Cross-entropy gradient: (predictions - targets) / n (for softmax + cross-entropy)

        // Compute (predictions - targets)
        let diff = predictions.sub(targets)?;

        // Compute (predictions - targets) / n
        let size = predictions.size();
        let n = NumCast::from(size).ok_or_else(|| TensorError::new("Failed to convert matrix size to numeric type"))?;
        Ok(diff.scalar_div(n))
    }
}