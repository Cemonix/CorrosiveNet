pub mod linear;
pub mod conv;

pub use crate::math::{Matrix, MatrixError};
pub use linear::Linear;
pub use conv::Conv2D;

pub trait Layer<T> {
    fn forward(&mut self, input: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
    fn backward(&mut self, grad_output: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
    fn parameters(&self) -> Vec<&Matrix<T>>;
    fn gradients(&self) -> Vec<&Matrix<T>>;
}