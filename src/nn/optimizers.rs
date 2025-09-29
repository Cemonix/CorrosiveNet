pub mod sgd;
pub mod adam;

pub use crate::math::{Tensor, TensorError};
pub use sgd::SGD;
pub use adam::Adam;

pub trait Optimizer<T> {
    fn step(&mut self, parameters: &mut [&mut Tensor<T>], gradients: &[&Tensor<T>]) -> Result<(), TensorError>;
    fn zero_grad(&mut self, gradients: &mut [&mut Tensor<T>]) -> Result<(), TensorError>;
}