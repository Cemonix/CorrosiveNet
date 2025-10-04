use cudarc::driver::CudaSlice;
use crate::cuda::{CudaBackend, CudaError};
use super::utils::{launch_binary_elementwise, launch_unary_elementwise, launch_scalar_elementwise};

/// Elementwise operation kernel launchers
///
/// Provides high-level functions to launch CUDA kernels for elementwise operations
/// on tensors. All operations follow GPU-parallel patterns where each thread processes
/// one element.
pub struct ElementwiseKernels;

impl ElementwiseKernels {
    // ==================== Binary Operations ====================

    /// Element-wise addition: c = a + b
    pub fn add_f32(
        backend: &CudaBackend,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_binary_elementwise(
            backend,
            "elementwise_add_f32",
            include_str!("../../kernels/elementwise/add.cu"),
            a, b, c, n
        )
    }

    /// Element-wise addition: c = a + b (f64)
    pub fn add_f64(
        backend: &CudaBackend,
        a: &CudaSlice<f64>,
        b: &CudaSlice<f64>,
        c: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_binary_elementwise(
            backend,
            "elementwise_add_f64",
            include_str!("../../kernels/elementwise/add.cu"),
            a, b, c, n
        )
    }

    /// Element-wise subtraction: c = a - b
    pub fn sub_f32(
        backend: &CudaBackend,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_binary_elementwise(
            backend,
            "elementwise_sub_f32",
            include_str!("../../kernels/elementwise/sub.cu"),
            a, b, c, n
        )
    }

    /// Element-wise subtraction: c = a - b (f64)
    pub fn sub_f64(
        backend: &CudaBackend,
        a: &CudaSlice<f64>,
        b: &CudaSlice<f64>,
        c: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_binary_elementwise(
            backend,
            "elementwise_sub_f64",
            include_str!("../../kernels/elementwise/sub.cu"),
            a, b, c, n
        )
    }

    /// Element-wise multiplication: c = a * b
    pub fn mul_f32(
        backend: &CudaBackend,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_binary_elementwise(
            backend,
            "elementwise_mul_f32",
            include_str!("../../kernels/elementwise/mul.cu"),
            a, b, c, n
        )
    }

    /// Element-wise multiplication: c = a * b (f64)
    pub fn mul_f64(
        backend: &CudaBackend,
        a: &CudaSlice<f64>,
        b: &CudaSlice<f64>,
        c: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_binary_elementwise(
            backend,
            "elementwise_mul_f64",
            include_str!("../../kernels/elementwise/mul.cu"),
            a, b, c, n
        )
    }

    /// Element-wise division: c = a / b
    pub fn div_f32(
        backend: &CudaBackend,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_binary_elementwise(
            backend,
            "elementwise_div_f32",
            include_str!("../../kernels/elementwise/div.cu"),
            a, b, c, n
        )
    }

    /// Element-wise division: c = a / b (f64)
    pub fn div_f64(
        backend: &CudaBackend,
        a: &CudaSlice<f64>,
        b: &CudaSlice<f64>,
        c: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_binary_elementwise(
            backend,
            "elementwise_div_f64",
            include_str!("../../kernels/elementwise/div.cu"),
            a, b, c, n
        )
    }

    // ==================== Unary Operations ====================

    /// Element-wise negation: output = -input
    pub fn neg_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_neg_f32",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise negation: output = -input (f64)
    pub fn neg_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_neg_f64",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise absolute value: output = |input|
    pub fn abs_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_abs_f32",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise absolute value: output = |input| (f64)
    pub fn abs_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_abs_f64",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise square root: output = √input
    pub fn sqrt_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_sqrt_f32",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise square root: output = √input (f64)
    pub fn sqrt_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_sqrt_f64",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise exponential: output = e^input
    pub fn exp_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_exp_f32",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise exponential: output = e^input (f64)
    pub fn exp_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_exp_f64",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise natural logarithm: output = ln(input)
    pub fn log_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_log_f32",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise natural logarithm: output = ln(input) (f64)
    pub fn log_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_log_f64",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise square: output = input²
    pub fn square_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_square_f32",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    /// Element-wise square: output = input² (f64)
    pub fn square_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_unary_elementwise(
            backend,
            "elementwise_square_f64",
            include_str!("../../kernels/elementwise/unary.cu"),
            input, output, n
        )
    }

    // ==================== Scalar Operations ====================

    /// Scalar addition: output = input + scalar
    pub fn add_scalar_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        scalar: f32,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_scalar_elementwise(
            backend,
            "scalar_add_f32",
            include_str!("../../kernels/elementwise/scalar.cu"),
            input, scalar, output, n
        )
    }

    /// Scalar addition: output = input + scalar (f64)
    pub fn add_scalar_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        scalar: f64,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_scalar_elementwise(
            backend,
            "scalar_add_f64",
            include_str!("../../kernels/elementwise/scalar.cu"),
            input, scalar, output, n
        )
    }

    /// Scalar subtraction: output = input - scalar
    pub fn sub_scalar_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        scalar: f32,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_scalar_elementwise(
            backend,
            "scalar_sub_f32",
            include_str!("../../kernels/elementwise/scalar.cu"),
            input, scalar, output, n
        )
    }

    /// Scalar subtraction: output = input - scalar (f64)
    pub fn sub_scalar_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        scalar: f64,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_scalar_elementwise(
            backend,
            "scalar_sub_f64",
            include_str!("../../kernels/elementwise/scalar.cu"),
            input, scalar, output, n
        )
    }

    /// Scalar multiplication: output = input * scalar
    pub fn mul_scalar_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        scalar: f32,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_scalar_elementwise(
            backend,
            "scalar_mul_f32",
            include_str!("../../kernels/elementwise/scalar.cu"),
            input, scalar, output, n
        )
    }

    /// Scalar multiplication: output = input * scalar (f64)
    pub fn mul_scalar_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        scalar: f64,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_scalar_elementwise(
            backend,
            "scalar_mul_f64",
            include_str!("../../kernels/elementwise/scalar.cu"),
            input, scalar, output, n
        )
    }

    /// Scalar division: output = input / scalar
    pub fn div_scalar_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        scalar: f32,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_scalar_elementwise(
            backend,
            "scalar_div_f32",
            include_str!("../../kernels/elementwise/scalar.cu"),
            input, scalar, output, n
        )
    }

    /// Scalar division: output = input / scalar (f64)
    pub fn div_scalar_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        scalar: f64,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_scalar_elementwise(
            backend,
            "scalar_div_f64",
            include_str!("../../kernels/elementwise/scalar.cu"),
            input, scalar, output, n
        )
    }

    /// Scalar power: output = input^scalar
    pub fn pow_scalar_f32(
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        scalar: f32,
        output: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_scalar_elementwise(
            backend,
            "scalar_pow_f32",
            include_str!("../../kernels/elementwise/scalar.cu"),
            input, scalar, output, n
        )
    }

    /// Scalar power: output = input^scalar (f64)
    pub fn pow_scalar_f64(
        backend: &CudaBackend,
        input: &CudaSlice<f64>,
        scalar: f64,
        output: &mut CudaSlice<f64>,
        n: usize,
    ) -> Result<(), CudaError> {
        launch_scalar_elementwise(
            backend,
            "scalar_pow_f64",
            include_str!("../../kernels/elementwise/scalar.cu"),
            input, scalar, output, n
        )
    }
}
