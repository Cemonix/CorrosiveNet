pub mod dim;
pub mod shape;
pub mod elementwise;
pub mod scalar;
pub mod stats;
pub mod ops;

pub use dim::Dims;
pub use elementwise::MatrixElementwise;
pub use ops::MatrixOps;
pub use scalar::MatrixScalar;
pub use shape::MatrixShape;
pub use stats::MatrixStats;

#[derive(Debug)]
pub struct MatrixError {
    details: String,
}

impl MatrixError {
    pub fn new(details: &str) -> Self {
        MatrixError {
            details: details.to_string(),
        }
    }
}

impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Matrix error: {}", self.details)
    }
}

#[derive(Clone)]
pub struct Matrix<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T> Matrix<T> {
    /// Create a new matrix filled with zeros.
    ///
    /// # Arguments
    /// * `shape` - The dimensions of the matrix
    ///
    /// # Returns
    /// A new matrix with all elements set to zero
    pub fn zeros(shape: Vec<usize>) -> Result<Self, MatrixError>
    where
        T: Clone + Default,
    {
        Self::init_checks(&shape)?;

        let size: usize = shape.iter().product();
        let strides = Self::calculate_strides(&shape);

        Ok(Matrix {
            data: vec![T::default(); size],
            shape,
            strides,
        })
    }

    /// Create a new matrix filled with ones.
    ///
    /// # Arguments
    /// * `shape` - The dimensions of the matrix
    ///
    /// # Returns
    /// A new matrix with all elements set to one
    /// 
    /// # Errors
    /// When shape contains zero dimensions
    pub fn ones(shape: Vec<usize>) -> Result<Self, MatrixError>
    where
        T: Clone + Default + From<u8>,
    {
        Self::init_checks(&shape)?;

        let size: usize = shape.iter().product();
        let strides = Self::calculate_strides(&shape);

        Ok(Matrix {
            data: vec![T::from(1u8); size],
            shape,
            strides,
        })
    }

    /// Create a matrix from existing data with the specified shape.
    ///
    /// # Arguments
    /// * `data` - Vector containing the matrix data in row-major order
    /// * `shape` - The dimensions of the matrix
    ///
    /// # Returns
    /// A new matrix containing the provided data
    /// 
    /// # Errors
    /// When data length does not match shape or shape contains zero dimensions
    pub fn from_data(data: Vec<T>, shape: Vec<usize>) -> Result<Self, MatrixError> {
        let expected_size: usize = shape.iter().product();

        if data.len() != expected_size {
            return Err(MatrixError::new("Data size does not match shape"));
        }

        Self::init_checks(&shape)?;

        Ok(Matrix {
            data,
            shape: shape.clone(),
            strides: Self::calculate_strides(&shape),
        })
    }
    
    /// Get the shape (dimensions) of the matrix.
    ///
    /// # Returns
    /// Slice containing the size of each dimension
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the matrix.
    ///
    /// # Returns
    /// Slice containing the stride for each dimension
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the total number of elements in the matrix.
    ///
    /// # Returns
    /// Total number of elements across all dimensions
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to an element at the specified indices.
    ///
    /// # Arguments
    /// * `indices` - Array of indices for each dimension
    ///
    /// # Returns
    /// Reference to the element at the specified position
    ///
    /// # Errors
    /// When indices are out of bounds or incorrect number of indices provided
    pub fn get(&self, indices: &[usize]) -> Result<&T, MatrixError> {
        let flat_index = self.calculate_index(indices)?;
        Ok(&self.data[flat_index])
    }

    /// Get a mutable reference to an element at the specified indices.
    ///
    /// # Arguments
    /// * `indices` - Array of indices for each dimension
    ///
    /// # Returns
    /// Mutable reference to the element at the specified position
    ///
    /// # Errors
    /// When indices are out of bounds or incorrect number of indices provided
    pub fn get_mut(&mut self, indices: &[usize]) -> Result<&mut T, MatrixError> {
        let flat_index = self.calculate_index(indices)?;
        Ok(&mut self.data[flat_index])
    }

    /// Set the value of an element at the specified indices.
    ///
    /// # Arguments
    /// * `indices` - Array of indices for each dimension
    /// * `value` - The value to set
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When indices are out of bounds or incorrect number of indices provided
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<(), MatrixError> {
        let slot = self.get_mut(indices)?;
        *slot = value;
        Ok(())
    }

    /// Fill the entire matrix with a specified value.
    ///
    /// # Arguments
    /// * `value` - The value to fill the matrix with
    /// 
    /// # Errors
    /// When the length of values does not match the matrix size
    pub fn fill(&mut self, values: &Vec<T>)
    where
        T: Copy,
    {
        if values.len() != self.size() {
            panic!("Fill values length does not match matrix size");
        }
        self.data.copy_from_slice(&values);
    }

    /// Fill the entire matrix with zeros.
    pub fn fill_zeros(&mut self)
    where
        T: Copy + Default,
    {
        self.data.fill(T::default());
    }

    /// Check if the matrix data is in contiguous memory layout.
    ///
    /// # Returns
    /// True if matrix is stored contiguously in memory
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = Self::calculate_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Calculate strides for row-major order.
    /// 
    /// # Arguments
    /// * `shape` - The dimensions of the matrix
    /// 
    /// # Returns
    /// A vector containing the stride for each dimension
    fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let n = shape.len();
        let mut strides = vec![1; n];
        for i in (0..n - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Calculate flat index from multi-dimensional indices.
    /// 
    /// # Arguments
    /// * `indices` - Array of indices for each dimension
    ///
    /// # Returns
    /// The corresponding flat index in the data vector
    /// 
    /// # Errors
    /// When indices are out of bounds or incorrect number of indices provided
    fn calculate_index(&self, indices: &[usize]) -> Result<usize, MatrixError> {
        if indices.len() != self.shape.len() {
            return Err(MatrixError::new("Incorrect number of indices"));
        }
        let mut flat_index = 0;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(MatrixError::new("Index out of bounds"));
            }
            flat_index += idx * self.strides[i];
        }
        Ok(flat_index)
    }

    /// Convert flat index to multi-dimensional indices.
    /// 
    /// # Arguments
    /// * `flat_index` - The flat index in the data vector
    ///
    /// # Returns
    /// A vector containing the multi-dimensional indices
    fn flat_to_indices(&self, flat_index: usize) -> Vec<usize> {
        let n = self.shape.len();
        let mut indices = vec![0; n];
        let mut remaining = flat_index;

        for i in 0..n {
            indices[i] = remaining / self.strides[i];
            remaining %= self.strides[i];
        }

        indices
    }

    /// Validate matrix initialization parameters.
    /// 
    /// # Arguments
    /// * `shape` - The dimensions of the matrix
    /// 
    /// # Returns
    /// Unit type on success
    /// 
    /// # Errors
    /// When the shape is invalid
    fn init_checks(shape: &[usize]) -> Result<(), MatrixError> {
        if shape.is_empty() {
            return Err(MatrixError::new("Shape cannot be empty"));
        }

        for &dim in shape {
            if dim == 0 {
                return Err(MatrixError::new("Matrix dimensions must be greater than 0"));
            }
        }

        Ok(())
    }

    /// Generic helper for element-wise operations between two matrices.
    /// 
    /// # Arguments
    /// * `other` - The other matrix to operate with
    /// * `op` - The binary operation to apply element-wise
    ///
    /// # Returns
    /// A new matrix containing the result of the element-wise operation
    /// 
    /// # Errors
    /// When shapes do not match
    pub(crate) fn elementwise_op<F>(&self, other: &Matrix<T>, op: F) -> Result<Matrix<T>, MatrixError>
    where
        F: Fn(T, T) -> T,
        T: Copy,
    {
        if self.shape != other.shape {
            return Err(MatrixError::new("Shapes do not match for operation"));
        }

        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| op(*a, *b))
            .collect();

        Ok(Matrix {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        })
    }

    /// Generic helper for in-place element-wise operations between two matrices.
    /// 
    /// # Arguments
    /// * `other` - The other matrix to operate with
    /// * `op` - The binary operation to apply element-wise
    /// 
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When shapes do not match
    pub(crate) fn elementwise_op_inplace<F>(&mut self, other: &Matrix<T>, op: F) -> Result<(), MatrixError>
    where
        F: Fn(T, T) -> T,
        T: Copy,
    {
        if self.shape != other.shape {
            return Err(MatrixError::new("Shapes do not match for operation"));
        }

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = op(*a, *b);
        }

        Ok(())
    }

    /// Generic helper for scalar operations.
    /// 
    /// # Arguments
    /// * `scalar` - The scalar value to operate with
    /// * `op` - The binary operation to apply element-wise
    ///
    /// # Returns
    /// A new matrix containing the result of the scalar operation
    pub(crate) fn scalar_op<F>(&self, scalar: T, op: F) -> Matrix<T>
    where
        F: Fn(T, T) -> T,
        T: Copy,
    {
        let data: Vec<T> = self.data.iter().map(|&x| op(x, scalar)).collect();
        Matrix {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

// Display implementation for Matrix visualization
impl<T> std::fmt::Display for Matrix<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix{:?}[\n", self.shape)?;

        if self.shape.len() == 1 {
            // Vector case
            write!(f, "  [")?;
            for (i, value) in self.data.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}", value)?;
            }
            write!(f, "]")?;
        } else if self.shape.len() == 2 {
            // Matrix case
            let rows = self.shape[0];
            let cols = self.shape[1];
            for row in 0..rows {
                write!(f, "  [")?;
                for col in 0..cols {
                    if col > 0 { write!(f, ", ")?; }
                    let idx = row * cols + col;
                    write!(f, "{:8.4}", self.data[idx])?;
                }
                write!(f, "]{}", if row < rows - 1 { ",\n" } else { "\n" })?;
            }
        } else {
            // Higher dimensions - just show shape and first few elements
            write!(f, "  [first 5 elements: ")?;
            for (i, value) in self.data.iter().take(5).enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}", value)?;
            }
            if self.data.len() > 5 {
                write!(f, ", ...")?;
            }
            write!(f, "]")?;
        }

        write!(f, "\n]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_constructor() {
        let matrix = Matrix::<f32>::zeros(vec![3, 4]).unwrap();
        assert_eq!(matrix.shape(), &[3, 4]);
        assert_eq!(matrix.size(), 12);
        assert_eq!(matrix.strides(), &[4, 1]);

        // Check all elements are zero
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(*matrix.get(&[i, j]).unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn test_ones_constructor() {
        let matrix = Matrix::<f32>::ones(vec![2, 3]).unwrap();
        assert_eq!(matrix.shape(), &[2, 3]);
        assert_eq!(matrix.size(), 6);

        // Check all elements are one
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(*matrix.get(&[i, j]).unwrap(), 1.0);
            }
        }
    }

    #[test]
    fn test_from_data_constructor() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = Matrix::<f32>::from_data(data, vec![2, 3]).unwrap();

        assert_eq!(matrix.shape(), &[2, 3]);
        assert_eq!(matrix.size(), 6);

        // Check data layout: [[1, 2, 3], [4, 5, 6]]
        assert_eq!(*matrix.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*matrix.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(*matrix.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(*matrix.get(&[1, 0]).unwrap(), 4.0);
        assert_eq!(*matrix.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(*matrix.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_get_set_operations() {
        let mut matrix = Matrix::<f32>::zeros(vec![2, 2]).unwrap();

        // Test set and get
        matrix.set(&[0, 1], 5.5).unwrap();
        assert_eq!(*matrix.get(&[0, 1]).unwrap(), 5.5);

        // Test get_mut
        {
            let value = matrix.get_mut(&[1, 0]).unwrap();
            *value = 3.14;
        }
        assert_eq!(*matrix.get(&[1, 0]).unwrap(), 3.14);
    }

    #[test]
    fn test_strides_calculation() {
        let matrix3d = Matrix::<f32>::zeros(vec![2, 3, 4]).unwrap();
        assert_eq!(matrix3d.strides(), &[12, 4, 1]); // 3*4=12, 4=4, 1=1

        let matrix4d = Matrix::<f32>::zeros(vec![2, 3, 4, 5]).unwrap();
        assert_eq!(matrix4d.strides(), &[60, 20, 5, 1]); // 3*4*5=60, 4*5=20, 5=5, 1=1
    }

    #[test]
    fn test_is_contiguous() {
        // Regular matrix should be contiguous
        let matrix = Matrix::<f32>::zeros(vec![3, 4]).unwrap();
        assert!(matrix.is_contiguous());
    }

    #[test]
    fn test_error_handling_invalid_shape() {
        // Test zero dimensions
        let result = Matrix::<f32>::zeros(vec![0, 5]);
        assert!(result.is_err());

        let result = Matrix::<f32>::zeros(vec![5, 0]);
        assert!(result.is_err());

        // 1D vectors should work
        let result = Matrix::<f32>::zeros(vec![5]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_handling_index_out_of_bounds() {
        let matrix = Matrix::<f32>::zeros(vec![2, 3]).unwrap();

        // Test out of bounds access
        let result = matrix.get(&[2, 0]); // Row 2 doesn't exist (0-indexed)
        assert!(result.is_err());

        let result = matrix.get(&[0, 3]); // Column 3 doesn't exist
        assert!(result.is_err());

        let result = matrix.get(&[1]); // Wrong number of indices
        assert!(result.is_err());
    }

    #[test]
    fn test_error_handling_from_data() {
        // Test mismatched data size
        let data = vec![1.0, 2.0, 3.0]; // 3 elements
        let result = Matrix::<f32>::from_data(data, vec![2, 3]); // Expects 6 elements
        assert!(result.is_err());
    }
}