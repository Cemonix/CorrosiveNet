pub mod mse;
pub mod cross_entropy;

pub use crate::math::{Tensor, TensorError};
pub use mse::MeanSquaredError;
pub use cross_entropy::CrossEntropyLoss;

pub trait LossFunction<T> {
    fn forward(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<T, TensorError>;
    fn backward(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
}