mod initializers;

pub mod prelude {
    // Re-export corrosive-tensor prelude
    pub use corrosive_tensor::prelude::*;

    // Re-export corrosive-nn types
    pub use crate::initializers::Initializer;
}