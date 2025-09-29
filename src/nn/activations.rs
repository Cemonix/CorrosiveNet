pub mod relu;
pub mod sigmoid;
pub mod tanh;

pub use crate::math::{Tensor, TensorError};
pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use tanh::Tanh;

pub trait Activation<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn backward(&self, input: &Tensor<T>, grad_output: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
}