// Greater than: output = (input > threshold) ? 1 : 0
extern "C" __global__ void comparison_gt_f32(
    const float* input,
    float threshold,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] > threshold) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void comparison_gt_f64(
    const double* input,
    double threshold,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] > threshold) ? 1.0 : 0.0;
    }
}

// Greater than or equal: output = (input >= threshold) ? 1 : 0
extern "C" __global__ void comparison_ge_f32(
    const float* input,
    float threshold,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] >= threshold) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void comparison_ge_f64(
    const double* input,
    double threshold,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] >= threshold) ? 1.0 : 0.0;
    }
}

// Less than: output = (input < threshold) ? 1 : 0
extern "C" __global__ void comparison_lt_f32(
    const float* input,
    float threshold,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] < threshold) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void comparison_lt_f64(
    const double* input,
    double threshold,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] < threshold) ? 1.0 : 0.0;
    }
}

// Less than or equal: output = (input <= threshold) ? 1 : 0
extern "C" __global__ void comparison_le_f32(
    const float* input,
    float threshold,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] <= threshold) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void comparison_le_f64(
    const double* input,
    double threshold,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] <= threshold) ? 1.0 : 0.0;
    }
}

// Equal: output = (input == threshold) ? 1 : 0
extern "C" __global__ void comparison_eq_f32(
    const float* input,
    float threshold,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] == threshold) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void comparison_eq_f64(
    const double* input,
    double threshold,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] == threshold) ? 1.0 : 0.0;
    }
}

// Not equal: output = (input != threshold) ? 1 : 0
extern "C" __global__ void comparison_ne_f32(
    const float* input,
    float threshold,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] != threshold) ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void comparison_ne_f64(
    const double* input,
    double threshold,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (input[idx] != threshold) ? 1.0 : 0.0;
    }
}

// Clip max: output = min(input, threshold)
extern "C" __global__ void clip_max_f32(
    const float* input,
    float threshold,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = (val > threshold) ? threshold : val;
    }
}

extern "C" __global__ void clip_max_f64(
    const double* input,
    double threshold,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = input[idx];
        output[idx] = (val > threshold) ? threshold : val;
    }
}

// Clip min: output = max(input, threshold)
extern "C" __global__ void clip_min_f32(
    const float* input,
    float threshold,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = (val < threshold) ? threshold : val;
    }
}

extern "C" __global__ void clip_min_f64(
    const double* input,
    double threshold,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = input[idx];
        output[idx] = (val < threshold) ? threshold : val;
    }
}

// Clip: output = max(min_val, min(input, max_val))
extern "C" __global__ void clip_f32(
    const float* input,
    float min_val,
    float max_val,
    float* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        if (val < min_val) {
            output[idx] = min_val;
        } else if (val > max_val) {
            output[idx] = max_val;
        } else {
            output[idx] = val;
        }
    }
}

extern "C" __global__ void clip_f64(
    const double* input,
    double min_val,
    double max_val,
    double* output,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = input[idx];
        if (val < min_val) {
            output[idx] = min_val;
        } else if (val > max_val) {
            output[idx] = max_val;
        } else {
            output[idx] = val;
        }
    }
}
