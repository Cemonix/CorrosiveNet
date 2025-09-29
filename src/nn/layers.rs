pub mod linear;
pub mod conv;

pub use crate::math::{Tensor, TensorError};
pub use linear::Linear;
pub use conv::Conv2D;

pub trait Layer<T> {
    fn forward(&mut self, input: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn backward(&mut self, grad_output: &Tensor<T>) -> Result<Tensor<T>, TensorError>;
    fn parameters(&self) -> Vec<&Tensor<T>>;
    fn gradients(&self) -> Vec<&Tensor<T>>;
}