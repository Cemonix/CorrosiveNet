use super::{Tensor, TensorError, TensorCore, TensorStorage, Device};
use super::dim::TensorDims;

pub trait TensorShape<T> {
    fn transpose(&self) -> Result<Tensor<T>, TensorError>;
    fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor<T>, TensorError>;
    fn squeeze(&mut self, dims: TensorDims) -> Result<(), TensorError>;
    fn unsqueeze(&mut self, dims: TensorDims) -> Result<(), TensorError>;
    fn permute(&self, axes: Vec<usize>) -> Result<Tensor<T>, TensorError>;
}

impl<T> TensorShape<T> for Tensor<T>
where
    T: Clone,
{
    /// Transpose a 2D tensor (swaps rows and columns).
    ///
    /// # Returns
    /// A new tensor with transposed dimensions
    ///
    /// # Errors
    /// When tensor is not 2D
    fn transpose(&self) -> Result<Tensor<T>, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::new(&format!(
                "Transpose only supports 2D matrices, got {}D with shape {:?}",
                self.shape.len(), self.shape
            )));
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(0, 1);
        new_strides.swap(0, 1);

        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// Reshape the tensor to a new shape with the same total number of elements.
    ///
    /// # Arguments
    /// * `new_shape` - The desired new shape dimensions
    ///
    /// # Returns
    /// A new tensor with the specified shape
    ///
    /// # Errors
    /// When total number of elements doesn't match or new shape contains zero dimensions
    fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor<T>, TensorError> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err(TensorError::new(&format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                self.size(), new_shape, new_size
            )));
        }

        for &dim in &new_shape {
            if dim == 0 {
                return Err(TensorError::new("New shape dimensions must be greater than 0"));
            }
        }

        if self.is_contiguous() {
            let new_strides = Tensor::<T>::calculate_strides(&new_shape);
            Ok(Tensor {
                storage: self.storage.clone(),
                shape: new_shape,
                strides: new_strides,
            })
        } else {
            // For non-contiguous tensors, need to reorder data
            // First get data to CPU (will handle CUDA tensors)
            let mut new_data = Vec::with_capacity(self.size());

            for i in 0..self.size() {
                let indices = self.flat_to_indices(i);
                new_data.push((*self.get(&indices)?).clone());
            }

            let new_strides = Tensor::<T>::calculate_strides(&new_shape);

            // Preserve device by creating on same device
            match self.device() {
                Device::CPU => Ok(Tensor {
                    storage: TensorStorage::CPU(new_data),
                    shape: new_shape,
                    strides: new_strides,
                }),
                cuda_device => {
                    let cpu_tensor = Tensor {
                        storage: TensorStorage::CPU(new_data),
                        shape: new_shape,
                        strides: new_strides,
                    };
                    cpu_tensor.to(cuda_device)
                }
            }
        }
    }

    /// Remove dimensions of size 1 from the tensor shape in-place.
    ///
    /// # Arguments
    /// * `dims` - Which dimensions to squeeze (All, Single dimension, or Multiple dimensions)
    ///
    /// # Returns
    /// Unit type on success
    ///
    /// # Errors
    /// When trying to squeeze dimensions that are not size 1 or would result in empty shape
    fn squeeze(&mut self, dims: TensorDims) -> Result<(), TensorError> {
        match dims {
            TensorDims::All => {
                let new_shape: Vec<usize> = self.shape.iter().filter(|&&dim| dim != 1).copied().collect();
                if new_shape.is_empty() {
                    return Err(TensorError::new("Cannot squeeze all dimensions - would result in empty shape"));
                }
                self.shape = new_shape;
                self.strides = Tensor::<T>::calculate_strides(&self.shape);
                Ok(())
            },
            TensorDims::Single(dim) => {
                if dim >= self.shape.len() {
                    return Err(TensorError::new(&format!(
                        "Dimension {} is out of bounds for Tensor with {} dimensions",
                        dim, self.shape.len()
                    )));
                }
                if self.shape[dim] != 1 {
                    return Err(TensorError::new(&format!(
                        "Cannot squeeze dimension {} with size {}, only dimensions of size 1 can be squeezed",
                        dim, self.shape[dim]
                    )));
                }
                self.shape.remove(dim);
                if self.shape.is_empty() {
                    return Err(TensorError::new("Cannot squeeze - would result in empty shape"));
                }
                self.strides = Tensor::<T>::calculate_strides(&self.shape);
                Ok(())
            },
            TensorDims::Multiple(dims_to_squeeze) => {
                for &dim in &dims_to_squeeze {
                    if dim >= self.shape.len() {
                        return Err(TensorError::new(&format!(
                            "Dimension {} is out of bounds for tensor with {} dimensions",
                            dim, self.shape.len()
                        )));
                    }
                    if self.shape[dim] != 1 {
                        return Err(TensorError::new(&format!(
                            "Cannot squeeze dimension {} with size {}, only dimensions of size 1 can be squeezed",
                            dim, self.shape[dim]
                        )));
                    }
                }

                let mut sorted_dims = dims_to_squeeze.clone();
                sorted_dims.sort_by(|a, b| b.cmp(a));

                for &dim in &sorted_dims {
                    self.shape.remove(dim);
                }

                if self.shape.is_empty() {
                    return Err(TensorError::new("Cannot squeeze - would result in empty shape"));
                }

                self.strides = Tensor::<T>::calculate_strides(&self.shape);
                Ok(())
            }
        }
    }

    /// Add dimensions of size 1 to the tensor shape in-place.
    ///
    /// # Arguments
    /// * `dims` - Which dimensions to unsqueeze (Single dimension or Multiple dimensions)
    /// 
    /// # Returns
    /// Unit type on success
    /// 
    /// # Errors
    /// When trying to unsqueeze out-of-bounds dimensions or duplicate dimensions
    fn unsqueeze(&mut self, dims: TensorDims) -> Result<(), TensorError> {
        match dims {
            TensorDims::All => {
                return Err(TensorError::new("Unsqueeze does not support Dims::All. Use Dims::Single or Dims::Multiple"));
            },
            TensorDims::Single(dim) => {
                let max_dim = self.shape.len();
                if dim > max_dim {
                    return Err(TensorError::new(&format!(
                        "Dimension {} is out of bounds for unsqueeze operation (max allowed: {})",
                        dim, max_dim
                    )));
                }

                self.shape.insert(dim, 1);
                self.strides = Tensor::<T>::calculate_strides(&self.shape);
                Ok(())
            },
            TensorDims::Multiple(dims_to_unsqueeze) => {
                let max_final_dim = self.shape.len() + dims_to_unsqueeze.len();

                for &dim in &dims_to_unsqueeze {
                    if dim > max_final_dim {
                        return Err(TensorError::new(&format!(
                            "Dimension {} is out of bounds for unsqueeze operation (max allowed after all insertions: {})",
                            dim, max_final_dim
                        )));
                    }
                }

                let mut sorted_dims = dims_to_unsqueeze.clone();
                sorted_dims.sort();
                for i in 1..sorted_dims.len() {
                    if sorted_dims[i] == sorted_dims[i-1] {
                        return Err(TensorError::new(&format!(
                            "Duplicate dimension {} in unsqueeze operation",
                            sorted_dims[i]
                        )));
                    }
                }

                for &dim in &sorted_dims {
                    self.shape.insert(dim, 1);
                }

                self.strides = Tensor::<T>::calculate_strides(&self.shape);
                Ok(())
            }
        }
    }

    /// Permute the dimensions of the tensor according to the given axis order.
    ///
    /// # Arguments
    /// * `axes` - A vector of axis indices specifying the new order of dimensions
    ///
    /// # Returns
    /// A new tensor with permuted dimensions
    ///
    /// # Errors
    /// When axes are out of bounds or contain duplicates
    fn permute(&self, axes: Vec<usize>) -> Result<Tensor<T>, TensorError> {
        let mut seen = vec![false; self.shape.len()];
        for &axis in &axes {
            if axis >= self.shape.len() {
                return Err(TensorError::new(&format!(
                    "Axis {} is out of bounds for tensor with {} dimensions", axis, self.shape.len()
                )));
            }
            if seen[axis] {
                return Err(TensorError::new(&format!(
                    "Axis {} appears multiple times in permutation", axis
                )));
            }
            seen[axis] = true;
        }

        let mut new_shape = vec![0; self.shape.len()];
        let mut new_strides = vec![0; self.strides.len()];

        for (i, &axis) in axes.iter().enumerate() {
            new_shape[i] = self.shape[axis];
            new_strides[i] = self.strides[axis];
        }

        Ok(Tensor {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Device, Tensor, TensorCore, TensorDims, TensorShape, TensorInit};

    #[test]
    fn test_transpose() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Device::CPU).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.shape(), &[3, 2]);

        // Original: [[1, 2, 3], [4, 5, 6]]
        // Transposed: [[1, 4], [2, 5], [3, 6]]
        assert_eq!(*transposed.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*transposed.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(*transposed.get(&[1, 0]).unwrap(), 2.0);
        assert_eq!(*transposed.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(*transposed.get(&[2, 0]).unwrap(), 3.0);
        assert_eq!(*transposed.get(&[2, 1]).unwrap(), 6.0);
    }

    #[test]
    fn test_permute_3d() {
        let tensor = Tensor::<f32>::from_data(
            (0..24).map(|x| x as f32).collect(),
            vec![2, 3, 4],
            Device::CPU
        ).unwrap();

        // Permute from [2, 3, 4] to [4, 2, 3] (axes [2, 0, 1])
        let permuted = tensor.permute(vec![2, 0, 1]).unwrap();
        assert_eq!(permuted.shape(), &[4, 2, 3]);

        // Check a few elements to ensure permutation worked
        assert_eq!(*tensor.get(&[0, 0, 0]).unwrap(), *permuted.get(&[0, 0, 0]).unwrap());
        assert_eq!(*tensor.get(&[1, 2, 3]).unwrap(), *permuted.get(&[3, 1, 2]).unwrap());
    }

    #[test]
    fn test_permute_swap_last_two_dims() {
        let tensor = Tensor::<f32>::zeros(vec![2, 3, 4], Device::CPU).unwrap();
        let swapped = tensor.permute(vec![0, 2, 1]).unwrap();

        assert_eq!(swapped.shape(), &[2, 4, 3]);
        assert_eq!(swapped.strides(), &[12, 1, 4]); // Strides should be permuted too
    }

    #[test]
    fn test_reshape_2d_to_1d() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Device::CPU).unwrap();
        let reshaped = tensor.reshape(vec![6]).unwrap();

        assert_eq!(reshaped.shape(), &[6]);

        // Elements should be in the same order
        for i in 0..6 {
            assert_eq!(*reshaped.get(&[i]).unwrap(), (i + 1) as f32);
        }
    }

    #[test]
    fn test_reshape_1d_to_2d() {
        // Create real 1D Tensor
        let tensor = Tensor::<f32>::from_data((1..=12).map(|x| x as f32).collect(), vec![12], Device::CPU).unwrap();
        let reshaped = tensor.reshape(vec![3, 4]).unwrap();

        assert_eq!(reshaped.shape(), &[3, 4]);

        // Check a few elements
        assert_eq!(*reshaped.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*reshaped.get(&[0, 3]).unwrap(), 4.0);
        assert_eq!(*reshaped.get(&[2, 3]).unwrap(), 12.0);
    }

    #[test]
    fn test_reshape_3d_to_2d() {
        let tensor = Tensor::<f32>::from_data(
            (1..=24).map(|x| x as f32).collect(),
            vec![2, 3, 4],
            Device::CPU
        ).unwrap();

        let reshaped = tensor.reshape(vec![6, 4]).unwrap();
        assert_eq!(reshaped.shape(), &[6, 4]);
        assert_eq!(reshaped.size(), 24);
    }

    #[test]
    fn test_squeeze_all_dimensions() {
        // Test squeezing all dimensions of size 1
        let mut tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2, 1], Device::CPU).unwrap();
        tensor.squeeze(TensorDims::All).unwrap();

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_squeeze_single_dimension() {
        let mut tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2], Device::CPU).unwrap();
        tensor.squeeze(TensorDims::Single(0)).unwrap();

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_squeeze_multiple_dimensions() {
        let mut tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 1, 2], Device::CPU).unwrap();
        tensor.squeeze(TensorDims::Multiple(vec![0, 2])).unwrap();

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_unsqueeze_single_dimension() {
        // Add dimension at the beginning
        let mut tensor1 = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        tensor1.unsqueeze(TensorDims::Single(0)).unwrap();
        assert_eq!(tensor1.shape(), &[1, 2, 2]);
        assert_eq!(*tensor1.get(&[0, 0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor1.get(&[0, 1, 1]).unwrap(), 4.0);

        // Add dimension at the end
        let mut tensor2 = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        tensor2.unsqueeze(TensorDims::Single(2)).unwrap();
        assert_eq!(tensor2.shape(), &[2, 2, 1]);
        assert_eq!(*tensor2.get(&[0, 0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor2.get(&[1, 1, 0]).unwrap(), 4.0);

        // Add dimension in the middle
        let mut tensor3 = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();
        tensor3.unsqueeze(TensorDims::Single(1)).unwrap();
        assert_eq!(tensor3.shape(), &[2, 1, 2]);
        assert_eq!(*tensor3.get(&[0, 0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor3.get(&[1, 0, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_unsqueeze_multiple_dimensions() {
        let mut tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();

        // Add multiple dimensions
        tensor.unsqueeze(TensorDims::Multiple(vec![0, 3])).unwrap();
        assert_eq!(tensor.shape(), &[1, 2, 2, 1]);
        assert_eq!(*tensor.get(&[0, 0, 0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get(&[0, 1, 1, 0]).unwrap(), 4.0);
    }

    #[test]
    fn test_squeeze_unsqueeze_roundtrip() {
        let original = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();

        // Add dimension and then remove it
        let mut test_tensor = original.clone();
        test_tensor.unsqueeze(TensorDims::Single(0)).unwrap();
        assert_eq!(test_tensor.shape(), &[1, 2, 2]);

        test_tensor.squeeze(TensorDims::Single(0)).unwrap();
        assert_eq!(test_tensor.shape(), original.shape());

        // Check that data is preserved
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    *original.get(&[i, j]).unwrap(),
                    *test_tensor.get(&[i, j]).unwrap()
                );
            }
        }
    }

    #[test]
    fn test_error_handling_transpose() {
        // Test 3D Tensor transpose (not supported)
        let tensor_3d = Tensor::<f32>::zeros(vec![2, 3, 4], Device::CPU).unwrap();
        let result = tensor_3d.transpose();
        assert!(result.is_err());
    }

    #[test]
    fn test_error_handling_permute() {
        let tensor = Tensor::<f32>::zeros(vec![2, 3, 4], Device::CPU).unwrap();

        // Test invalid axis (out of bounds)
        let result = tensor.permute(vec![0, 1, 3]); // Axis 3 doesn't exist
        assert!(result.is_err());

        // Test duplicate axis
        let result = tensor.permute(vec![0, 1, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_handling_reshape() {
        let tensor = Tensor::<f32>::zeros(vec![2, 3], Device::CPU).unwrap(); // 6 elements

        // Test incompatible size
        let result = tensor.reshape(vec![2, 4]); // 8 elements ≠ 6 elements
        assert!(result.is_err());

        // Test zero dimension in new shape
        let result = tensor.reshape(vec![0, 6]);
        assert!(result.is_err());
    }

    #[test]
    fn test_squeeze_error_cases() {
        let mut tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();

        // Try to squeeze dimension that isn't size 1
        let result = tensor.squeeze(TensorDims::Single(0));
        assert!(result.is_err());

        // Try to squeeze out of bounds dimension
        let result = tensor.squeeze(TensorDims::Single(5));
        assert!(result.is_err());

        // Try to squeeze all dimensions from a tensor that would become empty
        let mut scalar_tensor = Tensor::<f32>::from_data(vec![42.0], vec![1], Device::CPU).unwrap();
        let result = scalar_tensor.squeeze(TensorDims::All);
        assert!(result.is_err());
    }

    #[test]
    fn test_unsqueeze_error_cases() {
        let mut tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();

        // Try to use Dims::All (not supported)
        let result = tensor.unsqueeze(TensorDims::All);
        assert!(result.is_err());

        // Try to add dimension out of bounds
        let result = tensor.unsqueeze(TensorDims::Single(5));
        assert!(result.is_err());

        // Try to add duplicate dimensions
        let result = tensor.unsqueeze(TensorDims::Multiple(vec![0, 0]));
        assert!(result.is_err());
    }
}