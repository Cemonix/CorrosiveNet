use crate::{
    math::{Matrix, MatrixError},
    nn::{
        Layer, Optimizer, LossFunction, Activation,
        layers::Linear,
        optimizers::SGD,
        loss::{MeanSquaredError, CrossEntropyLoss},
        activations::{ReLU, Sigmoid, Tanh},
        initializers::{Initializer, InitializerType},
    },
};
use num_traits::NumCast;

/// PyTorch Lightning-style trainer for neural networks
pub struct Trainer<T> {
    /// Learning rate for optimization
    learning_rate: T,
    /// Maximum number of epochs to train
    max_epochs: usize,
    /// Print loss every N epochs (0 = no printing)
    log_every_n_epochs: usize,
}

impl<T> Trainer<T>
where
    T: Clone + Copy + std::fmt::Display,
{
    /// Create a new trainer with specified configuration
    ///
    /// # Arguments
    /// * `learning_rate` - Learning rate for the optimizer
    /// * `max_epochs` - Maximum number of training epochs
    /// * `log_every_n_epochs` - Print loss every N epochs (0 = no logging)
    ///
    /// # Returns
    /// A new Trainer instance
    pub fn new(learning_rate: T, max_epochs: usize, log_every_n_epochs: usize) -> Self {
        Trainer {
            learning_rate,
            max_epochs,
            log_every_n_epochs,
        }
    }

    /// Train a simple network with one linear layer and activation
    ///
    /// # Arguments
    /// * `layer` - The linear layer to train
    /// * `activation` - Activation function to use
    /// * `loss_fn` - Loss function to optimize
    /// * `optimizer` - Optimizer for parameter updates
    /// * `train_data` - Training input data
    /// * `train_targets` - Training target data
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn fit_simple<L, A, Loss, Opt>(
        &self,
        layer: &mut L,
        activation: &A,
        loss_fn: &Loss,
        optimizer: &mut Opt,
        train_data: &Matrix<T>,
        train_targets: &Matrix<T>,
    ) -> Result<(), MatrixError>
    where
        L: Layer<T>,
        A: Activation<T>,
        Loss: LossFunction<T>,
        Opt: Optimizer<T>,
        T: std::fmt::Display,
    {
        println!("Starting training for {} epochs...", self.max_epochs);

        for epoch in 0..self.max_epochs {
            // Forward pass
            let layer_output = layer.forward(train_data)?;
            let activated_output = activation.forward(&layer_output)?;

            // Compute loss
            let loss = loss_fn.forward(&activated_output, train_targets)?;

            // Backward pass
            let loss_grad = loss_fn.backward(&activated_output, train_targets)?;
            let activation_grad = activation.backward(&layer_output, &loss_grad)?;
            let _input_grad = layer.backward(&activation_grad)?;

            // Update parameters - we need mutable access to the layer's parameters
            // For now, let's implement a simpler approach specific to Linear layers
            // In a real implementation, we'd need mutable parameter access methods

            // TODO: This is a simplified approach - we need to redesign the Layer trait
            // to provide mutable parameter access for proper optimization

            // For now, let's just print that we would update parameters
            println!("  Would update {} parameters and zero {} gradients",
                     layer.parameters().len(), layer.gradients().len());

            // Log progress
            if self.log_every_n_epochs > 0 && epoch % self.log_every_n_epochs == 0 {
                println!("Epoch {}: Loss = {}", epoch, loss);
            }
        }

        println!("Training completed!");
        Ok(())
    }
}

/// Builder pattern for creating and configuring neural networks
pub struct NetworkBuilder<T> {
    input_size: usize,
    output_size: usize,
    initializer: InitializerType,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> NetworkBuilder<T>
where
    T: Clone + Default + Copy + NumCast,
{
    /// Start building a new network with specified input and output sizes
    pub fn new(input_size: usize, output_size: usize) -> Self {
        NetworkBuilder {
            input_size,
            output_size,
            initializer: InitializerType::XavierUniform,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the weight initializer
    pub fn with_initializer(mut self, initializer: InitializerType) -> Self {
        self.initializer = initializer;
        self
    }

    /// Build the linear layer with the specified configuration
    pub fn build_linear(self) -> Result<Linear<T>, MatrixError> {
        let mut layer = Linear::new(self.input_size, self.output_size)?;

        // TODO: Use the initializer to initialize weights
        // For now, we're using the default zeros initialization
        // We'll fix this in the next step!

        Ok(layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let trainer = Trainer::new(0.01f32, 100, 10);
        assert_eq!(trainer.learning_rate, 0.01f32);
        assert_eq!(trainer.max_epochs, 100);
        assert_eq!(trainer.log_every_n_epochs, 10);
    }

    #[test]
    fn test_network_builder() {
        let builder = NetworkBuilder::<f32>::new(2, 1);
        let layer = builder.build_linear().unwrap();

        assert_eq!(layer.weights().shape(), &[2, 1]);
        assert_eq!(layer.bias().shape(), &[1]);
    }
}