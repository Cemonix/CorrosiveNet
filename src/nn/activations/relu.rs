use num_traits::Num;

use crate::{math::{Tensor, TensorError, TensorElementwise}, nn::Activation};

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl<T> Activation<T> for ReLU
where
    T: Clone + Default + PartialOrd + From<u8> + Copy + Num,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        // Apply ReLU: max(0, x) using clip_min
        let zero = T::from(0u8);
        Ok(input.clip_min(zero))
    }

    fn backward(&self, input: &Tensor<T>, grad_output: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        // ReLU derivative: 1 if x > 0, else 0
        // grad_input = grad_output * (input > 0)

        let zero = T::from(0u8);

        // Create a mask: 1 where input > 0, 0 otherwise
        let mask = input.greater_than(zero);

        // Element-wise multiplication: grad_output * mask
        grad_output.elementwise_mul(&mask)
    }
}