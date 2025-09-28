use std::ops::{Add, AddAssign, Mul, Sub, Div};

use crate::{
    math::{Matrix, MatrixError, MatrixOps, MatrixShape, MatrixStats},
    nn::{Layer, initializers::Initializer}
};
use num_traits::NumCast;

pub struct Linear<T> {
    weights: Matrix<T>,
    bias: Matrix<T>,
    grad_weights: Matrix<T>,
    grad_bias: Matrix<T>,
    last_input: Option<Matrix<T>>,
}

impl<T> Linear<T>
where
    T: Clone + Default + Copy + NumCast,
{
    /// Creates a new Linear layer with default Xavier Uniform initialization
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `output_size` - Number of output features
    ///
    /// # Returns
    /// A new Linear layer with Xavier Uniform initialized weights and zero bias
    ///
    /// # Errors
    /// When matrix creation fails
    pub fn new(input_size: usize, output_size: usize) -> Result<Self, MatrixError> {
        Self::new_with_initializer(input_size, output_size, Initializer::xavier_uniform())
    }

    /// Creates a new Linear layer with custom initialization
    ///
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `output_size` - Number of output features
    /// * `initializer` - The initializer to use for weights
    ///
    /// # Returns
    /// A new Linear layer with custom initialized weights and zero bias
    ///
    /// # Errors
    /// When matrix creation or initialization fails
    pub fn new_with_initializer(input_size: usize, output_size: usize, initializer: Initializer) -> Result<Self, MatrixError>
    where
        T: Copy + NumCast,
    {
        let mut weights = Matrix::zeros(vec![input_size, output_size])?;
        let bias = Matrix::zeros(vec![output_size])?;
        let grad_weights = Matrix::zeros(vec![input_size, output_size])?;
        let grad_bias = Matrix::zeros(vec![output_size])?;

        // Initialize weights using the provided initializer
        initializer.initialize(&mut weights)?;

        Ok(Linear {
            weights,
            bias,
            grad_weights,
            grad_bias,
            last_input: None,
        })
    }

    pub fn weights(&self) -> &Matrix<T> {
        &self.weights
    }

    pub fn bias(&self) -> &Matrix<T> {
        &self.bias
    }

    pub fn weights_mut(&mut self) -> &mut Matrix<T> {
        &mut self.weights
    }

    pub fn bias_mut(&mut self) -> &mut Matrix<T> {
        &mut self.bias
    }

    pub fn weight_gradients(&self) -> &Matrix<T> {
        &self.grad_weights
    }

    pub fn bias_gradients(&self) -> &Matrix<T> {
        &self.grad_bias
    }
}

impl<T> Layer<T> for Linear<T>
where
    T: Clone + Default + Copy + NumCast + Add<Output = T> + Sub<Output = T> + AddAssign + Mul<Output = T> + Div<Output = T> + PartialOrd,
{
    fn forward(&mut self, input: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        // Store input for backward pass
        self.last_input = Some(input.clone());

        // Compute: output = input @ weights + bias
        let output = input.matmul(&self.weights)?;

        // Add bias using broadcasting
        output.broadcast_add(&self.bias)
    }

    fn backward(&mut self, grad_output: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if let Some(ref input) = self.last_input {
            // Compute gradient w.r.t. weights: grad_weights = input^T @ grad_output
            let input_t = input.transpose()?;
            self.grad_weights = input_t.matmul(grad_output)?;

            // Compute gradient w.r.t. bias: grad_bias = sum(grad_output, axis=0)
            // Sum across the batch dimension to get the bias gradient
            let bias_grad_sum = grad_output.sum()?;
            // Create a matrix with the same shape as bias containing the sum
            self.grad_bias = Matrix::from_data(vec![bias_grad_sum], vec![1])?;

            // Compute gradient w.r.t. input: grad_input = grad_output @ weights^T
            let weights_t = self.weights.transpose()?;
            grad_output.matmul(&weights_t)
        } else {
            Err(MatrixError::new("Cannot call backward without a forward pass"))
        }
    }

    fn parameters(&self) -> Vec<&Matrix<T>> {
        vec![&self.weights, &self.bias]
    }

    fn gradients(&self) -> Vec<&Matrix<T>> {
        vec![&self.grad_weights, &self.grad_bias]
    }
}