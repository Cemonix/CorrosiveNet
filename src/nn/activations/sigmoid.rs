use crate::{math::{Matrix, MatrixError, MatrixElementwise, MatrixScalar}, nn::Activation};
use num_traits::{Float, Num, One};

pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid
    }
}

impl<T> Activation<T> for Sigmoid
where
    T: Clone + Copy + Float + One + Default + From<u8> + Num + PartialOrd,
{
    fn forward(&self, input: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        // Apply sigmoid: 1 / (1 + exp(-x))

        // First compute -x
        let neg_input = input.scalar_mul(-T::one());

        // Then compute exp(-x)
        let exp_neg_input = neg_input.exp();

        // Then compute 1 + exp(-x)
        let ones = Matrix::ones(input.shape().to_vec())?;
        let one_plus_exp = ones.add(&exp_neg_input)?;

        // Finally compute 1 / (1 + exp(-x))
        let ones_again = Matrix::ones(input.shape().to_vec())?;
        ones_again.elementwise_div(&one_plus_exp)
    }

    fn backward(&self, input: &Matrix<T>, grad_output: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        // grad_input = grad_output * sigmoid(x) * (1 - sigmoid(x))

        // Compute sigmoid(x)
        let sigmoid_x = self.forward(input)?;

        // Compute (1 - sigmoid(x))
        let ones = Matrix::ones(input.shape().to_vec())?;
        let one_minus_sigmoid = ones.sub(&sigmoid_x)?;

        // Compute sigmoid(x) * (1 - sigmoid(x))
        let sigmoid_derivative = sigmoid_x.elementwise_mul(&one_minus_sigmoid)?;

        // Compute grad_output * sigmoid_derivative
        grad_output.elementwise_mul(&sigmoid_derivative)
    }
}