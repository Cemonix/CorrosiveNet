pub mod initializers;
pub mod parameter;

pub use parameter::Parameter;

pub mod prelude {
    // Re-export corrosive-tensor prelude
    pub use corrosive_tensor::prelude::*;

    // Re-export corrosive-nn types
    pub use crate::initializers::Initializer;
    pub use crate::parameter::Parameter;
}