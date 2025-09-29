use crate::{math::{Tensor, TensorError, TensorElementwise, TensorScalar}, nn::Activation};
use num_traits::{Float, Num, One};

pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Tanh
    }
}

impl<T> Activation<T> for Tanh
where
    T: Clone + Copy + Float + One + Default + From<u8> + Num + PartialOrd,
{
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        // Apply tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))

        // Compute exp(x)
        let exp_x = input.exp();

        // Compute exp(-x)
        let neg_input = input.scalar_mul(-T::one());
        let exp_neg_x = neg_input.exp();

        // Compute exp(x) - exp(-x)
        let numerator = exp_x.sub(&exp_neg_x)?;

        // Compute exp(x) + exp(-x)
        let denominator = exp_x.add(&exp_neg_x)?;

        // Compute (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        numerator.elementwise_div(&denominator)
    }

    fn backward(&self, input: &Tensor<T>, grad_output: &Tensor<T>) -> Result<Tensor<T>, TensorError> {
        // Tanh derivative: 1 - tanh^2(x)
        // grad_input = grad_output * (1 - tanh^2(x))

        // Compute tanh(x)
        let tanh_x = self.forward(input)?;

        // Compute tanh^2(x)
        let tanh_squared = tanh_x.square();

        // Compute (1 - tanh^2(x))
        let ones = Tensor::ones(input.shape().to_vec())?;
        let tanh_derivative = ones.sub(&tanh_squared)?;

        // Compute grad_output * tanh_derivative
        grad_output.elementwise_mul(&tanh_derivative)
    }
}