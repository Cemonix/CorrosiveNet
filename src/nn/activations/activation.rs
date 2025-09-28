use crate::math::{Matrix, MatrixError};

pub trait Activation<T> {
    fn forward(&self, input: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
    fn backward(&self, input: &Matrix<T>, grad_output: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
}