use num_traits::{Num, Float, Signed};
use std::fmt::{Display, LowerExp};

/// Base trait for all tensor element types
///
/// Provides the minimum requirements for storing and basic operations
pub trait TensorElement: Copy + Default + PartialOrd + Display + Send + Sync + 'static {}

/// Trait for numeric types suitable for tensor arithmetic operations
///
/// Extends TensorElement with basic numeric operations from num_traits
pub trait TensorNum: TensorElement + Num {}

/// Trait for signed numeric types
///
/// Adds sign-related operations for types that can be negative
pub trait TensorSigned: TensorNum + Signed {}

/// Trait for floating-point types suitable for tensor operations
///
/// Provides all operations needed for floating-point tensor computations
/// including scientific notation support and advanced math functions
pub trait TensorFloat: TensorSigned + Float + LowerExp {}

/// Trait for boolean types suitable for tensor mask operations
///
/// Provides logical operations needed for boolean tensor computations
/// including masking, filtering, and conditional operations
pub trait TensorBool: TensorElement {
    /// Logical AND operation
    fn and(self, other: Self) -> Self;

    /// Logical OR operation
    fn or(self, other: Self) -> Self;

    /// Logical NOT operation
    fn not(self) -> Self;

    /// Check if value is true (for counting, any/all operations)
    fn is_true(self) -> bool;
}

// Blanket implementations - automatically implement for compatible types

impl<T> TensorElement for T
where
    T: Copy + Default + PartialOrd + Display + Send + Sync + 'static
{}

impl<T> TensorNum for T
where
    T: TensorElement + Num
{}

impl<T> TensorSigned for T
where
    T: TensorNum + Signed
{}

impl<T> TensorFloat for T
where
    T: TensorSigned + Float + LowerExp
{}

impl TensorBool for bool {
    fn and(self, other: Self) -> Self {
        self && other
    }

    fn or(self, other: Self) -> Self {
        self || other
    }

    fn not(self) -> Self {
        !self
    }

    fn is_true(self) -> bool {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trait_implementations() {
        // These should compile if our traits work correctly
        fn check_tensor_element<T: TensorElement>() {}
        fn check_tensor_num<T: TensorNum>() {}
        fn check_tensor_signed<T: TensorSigned>() {}
        fn check_tensor_float<T: TensorFloat>() {}

        // Float types
        check_tensor_element::<f32>();
        check_tensor_num::<f32>();
        check_tensor_signed::<f32>();
        check_tensor_float::<f32>();

        check_tensor_element::<f64>();
        check_tensor_num::<f64>();
        check_tensor_signed::<f64>();
        check_tensor_float::<f64>();

        // Signed integer types
        check_tensor_element::<i32>();
        check_tensor_num::<i32>();
        check_tensor_signed::<i32>();

        check_tensor_element::<i64>();
        check_tensor_num::<i64>();
        check_tensor_signed::<i64>();

        // Unsigned types (no signing)
        check_tensor_element::<u32>();
        check_tensor_num::<u32>();

        check_tensor_element::<u64>();
        check_tensor_num::<u64>();

        // Boolean type
        check_tensor_element::<bool>();
        fn check_tensor_bool<T: TensorBool>() {}
        check_tensor_bool::<bool>();
    }

    #[test]
    fn test_tensor_bool_operations() {
        // Test AND operation
        assert_eq!(true.and(true), true);
        assert_eq!(true.and(false), false);
        assert_eq!(false.and(true), false);
        assert_eq!(false.and(false), false);

        // Test OR operation
        assert_eq!(true.or(true), true);
        assert_eq!(true.or(false), true);
        assert_eq!(false.or(true), true);
        assert_eq!(false.or(false), false);

        // Test NOT operation
        assert_eq!(true.not(), false);
        assert_eq!(false.not(), true);

        // Test is_true operation
        assert_eq!(true.is_true(), true);
        assert_eq!(false.is_true(), false);
    }
}