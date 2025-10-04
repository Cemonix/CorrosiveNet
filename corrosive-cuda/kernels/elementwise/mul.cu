extern "C" __global__ void elementwise_mul_f32(
    const float* a,
    const float* b,
    float* c,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

extern "C" __global__ void elementwise_mul_f64(
    const double* a,
    const double* b,
    double* c,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}
