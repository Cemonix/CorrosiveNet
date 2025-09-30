use super::{Tensor, TensorNum, TensorFloat};

pub trait TensorMath<T> {
    fn exp(&self) -> Tensor<T> where T: TensorFloat;
    fn log(&self) -> Tensor<T> where T: TensorFloat;
    fn sqrt(&self) -> Tensor<T> where T: TensorFloat;
    fn square(&self) -> Tensor<T>;
    fn abs(&self) -> Tensor<T> where T: TensorFloat;
    fn pow(&self, exponent: T) -> Tensor<T> where T: TensorFloat;
}

impl<T> TensorMath<T> for Tensor<T>
where
    T: TensorNum,
{
    /// Element-wise exponential of the tensor.
    ///
    /// # Returns
    /// A new tensor with the exponential of each element
    fn exp(&self) -> Tensor<T>
    where
        T: TensorFloat,
    {
        self.unary_op(|x| x.exp())
    }

    /// Element-wise natural logarithm of the tensor.
    ///
    /// # Returns
    /// A new tensor with the natural logarithm of each element
    fn log(&self) -> Tensor<T>
    where
        T: TensorFloat,
    {
        self.unary_op(|x| x.ln())
    }

    /// Element-wise square root of the tensor.
    ///
    /// # Returns
    /// A new tensor with the square root of each element
    fn sqrt(&self) -> Tensor<T>
    where
        T: TensorFloat,
    {
        self.unary_op(|x| x.sqrt())
    }

    /// Element-wise square of the tensor.
    ///
    /// # Returns
    /// A new tensor with the square of each element
    fn square(&self) -> Tensor<T> {
        self.unary_op(|x| x * x)
    }

    /// Element-wise absolute value of the tensor.
    ///
    /// # Returns
    /// A new tensor with the absolute value of each element
    fn abs(&self) -> Tensor<T>
    where
        T: TensorFloat,
    {
        self.unary_op(|x| x.abs())
    }

    /// Element-wise power operation on the tensor.
    ///
    /// # Arguments
    /// * `exponent` - The exponent to raise each element to
    ///
    /// # Returns
    /// A new tensor with each element raised to the given power
    fn pow(&self, exponent: T) -> Tensor<T>
    where
        T: TensorFloat,
    {
        self.unary_op(|x| x.powf(exponent))
    }
}

impl<T> Tensor<T> {
    /// Generic helper for unary operations on tensor elements.
    ///
    /// # Arguments
    /// * `op` - The unary operation to apply to each element
    ///
    /// # Returns
    /// A new tensor with the operation applied to all elements
    fn unary_op<F>(&self, op: F) -> Tensor<T>
    where
        F: Fn(T) -> T,
        T: Copy,
    {
        let data: Vec<T> = self.data.iter().map(|&x| op(x)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: Self::calculate_strides(&self.shape),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Tensor, TensorCore, TensorStorage, TensorMath};


    #[test]
    fn test_square() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = tensor.square();

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0);  // 1^2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 4.0);  // 2^2
        assert_eq!(*result.get(&[1, 0]).unwrap(), 9.0);  // 3^2
        assert_eq!(*result.get(&[1, 1]).unwrap(), 16.0); // 4^2
    }

    #[test]
    fn test_sqrt() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 4.0, 9.0, 16.0], vec![2, 2]).unwrap();
        let result = tensor.sqrt();

        assert_eq!(*result.get(&[0, 0]).unwrap(), 1.0); // sqrt(1)
        assert_eq!(*result.get(&[0, 1]).unwrap(), 2.0); // sqrt(4)
        assert_eq!(*result.get(&[1, 0]).unwrap(), 3.0); // sqrt(9)
        assert_eq!(*result.get(&[1, 1]).unwrap(), 4.0); // sqrt(16)
    }

    #[test]
    fn test_exp() {
        let tensor = Tensor::<f32>::from_data(vec![0.0, 1.0], vec![1, 2]).unwrap();
        let result = tensor.exp();

        assert!((result.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-6); // exp(0) ≈ 1
        assert!((result.get(&[0, 1]).unwrap() - 2.71828).abs() < 1e-4); // exp(1) ≈ e
    }

    #[test]
    fn test_log() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, std::f32::consts::E], vec![1, 2]).unwrap();
        let result = tensor.log();

        assert!((result.get(&[0, 0]).unwrap() - 0.0).abs() < 1e-6); // ln(1) = 0
        assert!((result.get(&[0, 1]).unwrap() - 1.0).abs() < 1e-6); // ln(e) = 1
    }

    #[test]
    fn test_abs() {
        let tensor = Tensor::<f32>::from_data(vec![-2.0, -1.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let result = tensor.abs();

        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0); // |-2|
        assert_eq!(*result.get(&[0, 1]).unwrap(), 1.0); // |-1|
        assert_eq!(*result.get(&[1, 0]).unwrap(), 0.0); // |0|
        assert_eq!(*result.get(&[1, 1]).unwrap(), 1.0); // |1|
    }

    #[test]
    fn test_pow() {
        let tensor = Tensor::<f32>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
        let result = tensor.pow(3.0);

        assert_eq!(*result.get(&[0, 0]).unwrap(), 8.0);   // 2^3
        assert_eq!(*result.get(&[0, 1]).unwrap(), 27.0);  // 3^3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 64.0);  // 4^3
        assert_eq!(*result.get(&[1, 1]).unwrap(), 125.0); // 5^3
    }
}