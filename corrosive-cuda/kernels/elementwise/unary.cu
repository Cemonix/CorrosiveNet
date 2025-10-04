// Negation
extern "C" __global__ void elementwise_neg_f32(
    const float* input,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = -input[idx];
    }
}

extern "C" __global__ void elementwise_neg_f64(
    const double* input,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = -input[idx];
    }
}

// Absolute value
extern "C" __global__ void elementwise_abs_f32(
    const float* input,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fabsf(input[idx]);
    }
}

extern "C" __global__ void elementwise_abs_f64(
    const double* input,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fabs(input[idx]);
    }
}

// Square root
extern "C" __global__ void elementwise_sqrt_f32(
    const float* input,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrtf(input[idx]);
    }
}

extern "C" __global__ void elementwise_sqrt_f64(
    const double* input,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrt(input[idx]);
    }
}

// Exponential
extern "C" __global__ void elementwise_exp_f32(
    const float* input,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = expf(input[idx]);
    }
}

extern "C" __global__ void elementwise_exp_f64(
    const double* input,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = exp(input[idx]);
    }
}

// Natural logarithm
extern "C" __global__ void elementwise_log_f32(
    const float* input,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = logf(input[idx]);
    }
}

extern "C" __global__ void elementwise_log_f64(
    const double* input,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = log(input[idx]);
    }
}

// Square (x^2)
extern "C" __global__ void elementwise_square_f32(
    const float* input,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x * x;
    }
}

extern "C" __global__ void elementwise_square_f64(
    const double* input,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x = input[idx];
        output[idx] = x * x;
    }
}
