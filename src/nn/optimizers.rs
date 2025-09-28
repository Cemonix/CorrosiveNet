pub mod sgd;
pub mod adam;

pub use crate::math::{Matrix, MatrixError};
pub use sgd::SGD;
pub use adam::Adam;

pub trait Optimizer<T> {
    fn step(&mut self, parameters: &mut [&mut Matrix<T>], gradients: &[&Matrix<T>]) -> Result<(), MatrixError>;
    fn zero_grad(&mut self, gradients: &mut [&mut Matrix<T>]) -> Result<(), MatrixError>;
}