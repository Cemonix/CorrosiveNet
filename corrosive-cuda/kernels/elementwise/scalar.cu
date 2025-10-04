// Scalar addition: tensor + scalar
extern "C" __global__ void scalar_add_f32(
    const float* input,
    float scalar,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + scalar;
    }
}

extern "C" __global__ void scalar_add_f64(
    const double* input,
    double scalar,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] + scalar;
    }
}

// Scalar subtraction: tensor - scalar
extern "C" __global__ void scalar_sub_f32(
    const float* input,
    float scalar,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] - scalar;
    }
}

extern "C" __global__ void scalar_sub_f64(
    const double* input,
    double scalar,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] - scalar;
    }
}

// Scalar multiplication: tensor * scalar
extern "C" __global__ void scalar_mul_f32(
    const float* input,
    float scalar,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scalar;
    }
}

extern "C" __global__ void scalar_mul_f64(
    const double* input,
    double scalar,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scalar;
    }
}

// Scalar division: tensor / scalar
extern "C" __global__ void scalar_div_f32(
    const float* input,
    float scalar,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] / scalar;
    }
}

extern "C" __global__ void scalar_div_f64(
    const double* input,
    double scalar,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] / scalar;
    }
}

// Scalar power: tensor^scalar
extern "C" __global__ void scalar_pow_f32(
    const float* input,
    float scalar,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = powf(input[idx], scalar);
    }
}

extern "C" __global__ void scalar_pow_f64(
    const double* input,
    double scalar,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = pow(input[idx], scalar);
    }
}
