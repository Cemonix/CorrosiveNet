pub mod mse;
pub mod cross_entropy;

pub use crate::math::{Matrix, MatrixError};
pub use mse::MeanSquaredError;
pub use cross_entropy::CrossEntropyLoss;

pub trait LossFunction<T> {
    fn forward(&self, predictions: &Matrix<T>, targets: &Matrix<T>) -> Result<T, MatrixError>;
    fn backward(&self, predictions: &Matrix<T>, targets: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
}