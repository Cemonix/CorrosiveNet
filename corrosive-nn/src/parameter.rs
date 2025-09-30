use corrosive_tensor::prelude::*;
use std::ops::{Deref, DerefMut};

/// A learnable parameter that wraps a Tensor and tracks gradients.
///
/// Parameters are the core building blocks of neural networks, representing
/// weights and biases that are updated during training through backpropagation.
#[derive(Clone)]
pub struct Parameter<T> {
    data: Tensor<T>,
    grad: Option<Tensor<T>>,
    requires_grad: bool,
}

impl<T: TensorElement> Parameter<T> {
    /// Create a new parameter from a tensor.
    ///
    /// # Arguments
    /// * `data` - The tensor data for this parameter
    ///
    /// # Returns
    /// A new parameter with gradient tracking enabled
    pub fn new(data: Tensor<T>) -> Self {
        Parameter {
            data,
            grad: None,
            requires_grad: true,
        }
    }

    /// Create a parameter with gradient tracking disabled.
    ///
    /// # Arguments
    /// * `data` - The tensor data for this parameter
    ///
    /// # Returns
    /// A new parameter that won't accumulate gradients
    pub fn from_tensor_no_grad(data: Tensor<T>) -> Self {
        Parameter {
            data,
            grad: None,
            requires_grad: false,
        }
    }

    /// Get a reference to the parameter's data tensor.
    ///
    /// # Returns
    /// Reference to the underlying tensor
    pub fn data(&self) -> &Tensor<T> {
        &self.data
    }

    /// Get a mutable reference to the parameter's data tensor.
    ///
    /// # Returns
    /// Mutable reference to the underlying tensor
    pub fn data_mut(&mut self) -> &mut Tensor<T> {
        &mut self.data
    }

    /// Get a reference to the accumulated gradient.
    ///
    /// # Returns
    /// Optional reference to the gradient tensor (None if no gradient computed yet)
    pub fn grad(&self) -> Option<&Tensor<T>> {
        self.grad.as_ref()
    }

    /// Get a mutable reference to the accumulated gradient.
    ///
    /// # Returns
    /// Optional mutable reference to the gradient tensor
    pub fn grad_mut(&mut self) -> Option<&mut Tensor<T>> {
        self.grad.as_mut()
    }

    /// Check if this parameter requires gradient computation.
    ///
    /// # Returns
    /// True if gradients will be tracked for this parameter
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set whether this parameter requires gradient computation.
    ///
    /// # Arguments
    /// * `requires_grad` - Whether to track gradients
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Zero out the accumulated gradients.
    ///
    /// This should be called before each training step to clear gradients
    /// from the previous backward pass.
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Accumulate gradient from a backward pass.
    ///
    /// Gradients are added together across multiple backward passes,
    /// which is useful for gradient accumulation across mini-batches.
    ///
    /// # Arguments
    /// * `grad` - Gradient tensor to accumulate
    ///
    /// # Returns
    /// Result indicating success or error
    pub fn accumulate_grad(&mut self, grad: Tensor<T>) -> Result<(), TensorError>
    where
        T: TensorNum,
    {
        if !self.requires_grad {
            return Ok(());
        }

        match &mut self.grad {
            Some(existing_grad) => {
                // Add new gradient to existing
                *existing_grad = existing_grad.add(&grad)?;
            }
            None => {
                // First gradient
                self.grad = Some(grad);
            }
        }

        Ok(())
    }

    /// Consume the parameter and return the inner tensor.
    ///
    /// # Returns
    /// The underlying tensor data
    pub fn into_inner(self) -> Tensor<T> {
        self.data
    }
}

impl<T: TensorNum> Parameter<T> {
    /// Create a parameter initialized with zeros.
    ///
    /// # Arguments
    /// * `shape` - Shape of the parameter tensor
    ///
    /// # Returns
    /// A new parameter filled with zeros
    ///
    /// # Panics
    /// If the shape is invalid
    pub fn zeros(shape: &[usize]) -> Self {
        Parameter::new(Tensor::zeros(shape.to_vec()).expect("Invalid shape for zeros"))
    }

    /// Create a parameter initialized with ones.
    ///
    /// # Arguments
    /// * `shape` - Shape of the parameter tensor
    ///
    /// # Returns
    /// A new parameter filled with ones
    ///
    /// # Panics
    /// If the shape is invalid
    pub fn ones(shape: &[usize]) -> Self {
        Parameter::new(Tensor::ones(shape.to_vec()).expect("Invalid shape for ones"))
    }
}

// Allow using Parameter like a Tensor for read operations
impl<T> Deref for Parameter<T> {
    type Target = Tensor<T>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

// Allow mutable access to underlying tensor
impl<T> DerefMut for Parameter<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
