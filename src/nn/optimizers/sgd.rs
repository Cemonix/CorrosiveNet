use std::ops::{Add, AddAssign, Div, Mul, Sub};

use crate::math::{Tensor, TensorError, TensorScalar, TensorOps};
use super::Optimizer;

pub struct SGD<T> {
    learning_rate: T,
}

impl<T> SGD<T>
where
    T: Clone + Copy,
{
    pub fn new(learning_rate: T) -> Self {
        SGD { learning_rate }
    }
}

impl<T> Optimizer<T> for SGD<T>
where
    T: Clone + Copy + Default + Add<Output = T> + Sub<Output = T> + AddAssign + Mul<Output = T> + Div<Output = T>,
{
    fn step(&mut self, parameters: &mut [&mut Tensor<T>], gradients: &[&Tensor<T>]) -> Result<(), TensorError> {
        if parameters.len() != gradients.len() {
            return Err(TensorError::new("Parameters and gradients must have the same length"));
        }

        for (param, grad) in parameters.iter_mut().zip(gradients.iter()) {
            // param = param - learning_rate * grad
            let scaled_grad = grad.scalar_mul(self.learning_rate);
            param.sub_inplace(&scaled_grad)?;
        }

        Ok(())
    }

    fn zero_grad(&mut self, gradients: &mut [&mut Tensor<T>]) -> Result<(), TensorError> {
        for grad in gradients.iter_mut() {
            // Set all gradients to zero
            grad.fill_zeros();
        }
        Ok(())
    }
}