mod math;
mod nn;
mod trainer;

use math::{Tensor, TensorOps, TensorScalar};
use nn::{
    layers::Linear,
    activations::ReLU,
    loss::MeanSquaredError,
    optimizers::SGD,
    Layer, Activation, LossFunction, Optimizer,
};

fn main() -> Result<(), math::TensorError> {
    println!("🚀 CorrosiveNet - First Neural Network Training!");

    // Create simple regression data: y = 2*x + 1
    let input_data = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1])?;
    let target_data = Tensor::<f32>::from_data(vec![3.0, 5.0, 7.0, 9.0], vec![4, 1])?;

    // Create network components
    let mut layer = Linear::<f32>::new(1, 1)?; // 1 input -> 1 output
    let activation = ReLU::new();
    let loss_fn = MeanSquaredError::new();
    let optimizer = SGD::new(0.01f32);

    println!("📊 Training data:");
    println!("Input:\n{}", input_data);
    println!("Target:\n{}", target_data);

    // Simple training loop
    let epochs = 160;
    for epoch in 0..epochs {
        // Forward pass
        let layer_output = layer.forward(&input_data)?;
        let activated_output = activation.forward(&layer_output)?;

        // Compute loss
        let loss = loss_fn.forward(&activated_output, &target_data)?;

        // Backward pass
        let loss_grad = loss_fn.backward(&activated_output, &target_data)?;
        let activation_grad = activation.backward(&layer_output, &loss_grad)?;
        let _input_grad = layer.backward(&activation_grad)?;

        // Print progress
        if epoch % 2 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss);
            println!("Weights:\n{}", layer.weights());
            println!("Bias:\n{}", layer.bias());
        }

        // Update parameters using optimizer
        // We need to do this separately to avoid borrowing conflicts

        // Update weights
        let weight_grad = layer.weight_gradients().clone();
        let scaled_weight_grad = weight_grad.scalar_mul(0.01f32);
        layer.weights_mut().sub_inplace(&scaled_weight_grad)?;

        // Update bias
        let bias_grad = layer.bias_gradients().clone();
        let scaled_bias_grad = bias_grad.scalar_mul(0.01f32);
        layer.bias_mut().sub_inplace(&scaled_bias_grad)?;
    }

    println!("✅ Training completed! Network architecture works!");
    Ok(())
}
