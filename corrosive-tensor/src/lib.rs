mod tensor;

// Core types always available at root
pub use tensor::{Tensor, TensorError};

// Traits available for explicit import
pub use tensor::{
    TensorElement, TensorNum, TensorSigned, TensorFloat, TensorBool,
    TensorCore, TensorStorage, TensorDims, TensorScalar, TensorShape,
    TensorStats, TensorMask, TensorLinAlg, TensorArithmetic, TensorMath,
    TensorComparison, TensorBroadcast,
};

// Prelude for convenience
pub mod prelude {
    pub use crate::tensor::{
        Tensor, TensorError,
        TensorElement, TensorNum, TensorSigned, TensorFloat, TensorBool,
        TensorCore, TensorStorage, TensorDims, TensorScalar, TensorShape,
        TensorStats, TensorMask, TensorLinAlg, TensorArithmetic, TensorMath,
        TensorComparison, TensorBroadcast,
    };
}
