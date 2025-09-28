pub mod relu;
pub mod sigmoid;
pub mod tanh;

pub use crate::math::{Matrix, MatrixError};
pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use tanh::Tanh;

pub trait Activation<T> {
    fn forward(&self, input: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
    fn backward(&self, input: &Matrix<T>, grad_output: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
}