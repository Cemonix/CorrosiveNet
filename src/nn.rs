pub mod layers;
pub mod activations;
pub mod loss;
pub mod optimizers;
pub mod initializers;

// Re-export main traits
pub use layers::Layer;
pub use activations::Activation;
pub use loss::LossFunction;
pub use optimizers::Optimizer;

// Re-export commonly used types
pub use layers::{Linear, Conv2D};
pub use activations::{ReLU, Sigmoid, Tanh};
pub use loss::{MeanSquaredError, CrossEntropyLoss};
pub use optimizers::{SGD, Adam};
pub use initializers::{Initializer, InitializerType};