use crate::math::{Matrix, MatrixError};

pub trait LossFunction<T> {
    fn forward(&self, predictions: &Matrix<T>, targets: &Matrix<T>) -> Result<T, MatrixError>;
    fn backward(&self, predictions: &Matrix<T>, targets: &Matrix<T>) -> Result<Matrix<T>, MatrixError>;
}