use super::{Tensor, TensorError, TensorCore, TensorNum, TensorStorage};
use super::dim::TensorDims;

pub trait TensorStats<T> {
    fn sum(&self) -> Result<T, TensorError>;
    fn mean(&self) -> Result<T, TensorError> where T: From<usize>;
    fn max(&self) -> Result<T, TensorError>;
    fn min(&self) -> Result<T, TensorError>;
    fn sum_dim(&self, dims: TensorDims, keepdim: bool) -> Result<Tensor<T>, TensorError>;
    fn mean_dim(&self, dims: TensorDims, keepdim: bool) -> Result<Tensor<T>, TensorError> where T: From<usize>;
    fn max_dim(&self, dims: TensorDims, keepdim: bool) -> Result<Tensor<T>, TensorError>;
    fn min_dim(&self, dims: TensorDims, keepdim: bool) -> Result<Tensor<T>, TensorError>;
}

impl<T> TensorStats<T> for Tensor<T>
where
    T: TensorNum,
{
    /// Calculate the sum of all elements in the tensor.
    /// 
    /// # Returns
    /// The sum of all elements as type T.
    /// 
    /// # Errors
    /// Returns TensorError if the tensor is empty.
    fn sum(&self) -> Result<T, TensorError> {
        Ok(self.data.iter().copied().fold(T::zero(), |acc, x| acc + x))
    }

    /// Calculate the mean (average) of all elements in the tensor.
    ///
    /// # Returns
    /// The mean of all elements as type T.
    ///
    /// # Errors
    /// Returns TensorError if the tensor is empty.
    fn mean(&self) -> Result<T, TensorError>
    where
        T: From<usize>,
    {
        if self.data.is_empty() {
            return Err(TensorError::new("Cannot compute mean of empty tensor"));
        }

        let sum = self.sum()?;
        let count = self.size();
        Ok(sum / T::from(count))
    }

    /// Calculate the minimum value in the tensor.
    ///
    /// # Returns
    /// The smallest element in the tensor
    ///
    /// # Errors
    /// When the tensor is empty
    fn min(&self) -> Result<T, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError::new("Cannot compute min of empty tensor"));
        }

        let mut min_val = self.data[0];
        for &val in &self.data[1..] {
            if val < min_val {
                min_val = val;
            }
        }
        Ok(min_val)
    }

    /// Calculate the maximum value in the tensor.
    ///
    /// # Returns
    /// The largest element in the tensor
    ///
    /// # Errors
    /// When the tensor is empty
    fn max(&self) -> Result<T, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError::new("Cannot compute max of empty tensor"));
        }

        let mut max_val = self.data[0];
        for &val in &self.data[1..] {
            if val > max_val {
                max_val = val;
            }
        }
        Ok(max_val)
    }

    /// Calculate the sum along specified dimensions.
    ///
    /// # Arguments
    /// * `dims` - Which dimensions to reduce (All, Single, or Multiple)
    /// * `keepdim` - Whether to keep reduced dimensions with size 1
    ///
    /// # Returns
    /// A new tensor with specified dimensions reduced
    ///
    /// # Errors
    /// When dimensions are out of bounds or tensor is empty
    fn sum_dim(&self, dims: TensorDims, keepdim: bool) -> Result<Tensor<T>, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError::new("Cannot compute sum_dim of empty tensor"));
        }

        let dims_to_reduce = self.normalize_dims(&dims)?;
        let output_shape = self.compute_output_shape(&dims_to_reduce, keepdim);
        let output_size: usize = output_shape.iter().product();
        let mut output_data = vec![T::zero(); output_size];

        for (out_idx, out_val) in output_data.iter_mut().enumerate() {
            let base_indices = self.output_idx_to_indices(out_idx, &output_shape, &dims_to_reduce, keepdim);
            *out_val = self.reduce_over_dims(&base_indices, &dims_to_reduce, |acc, val| acc + val, T::zero())?;
        }

        Tensor::from_data(output_data, output_shape, self.device.clone())
    }

    /// Calculate the mean along specified dimensions.
    ///
    /// # Arguments
    /// * `dims` - Which dimensions to reduce (All, Single, or Multiple)
    /// * `keepdim` - Whether to keep reduced dimensions with size 1
    ///
    /// # Returns
    /// A new tensor with specified dimensions reduced
    ///
    /// # Errors
    /// When dimensions are out of bounds or tensor is empty
    fn mean_dim(&self, dims: TensorDims, keepdim: bool) -> Result<Tensor<T>, TensorError>
    where
        T: From<usize>,
    {
        if self.data.is_empty() {
            return Err(TensorError::new("Cannot compute mean_dim of empty tensor"));
        }

        let dims_to_reduce = self.normalize_dims(&dims)?;
        let count: usize = dims_to_reduce.iter().map(|&d| self.shape[d]).product();
        let sum_result = self.sum_dim(dims, keepdim)?;

        let divisor = T::from(count);
        let mean_data: Vec<T> = sum_result.data.iter().map(|&x| x / divisor).collect();

        Tensor::from_data(mean_data, sum_result.shape, self.device.clone())
    }

    /// Calculate the maximum value along specified dimensions.
    ///
    /// # Arguments
    /// * `dims` - Which dimensions to reduce (All, Single, or Multiple)
    /// * `keepdim` - Whether to keep reduced dimensions with size 1
    ///
    /// # Returns
    /// A new tensor with specified dimensions reduced
    ///
    /// # Errors
    /// When dimensions are out of bounds or tensor is empty
    fn max_dim(&self, dims: TensorDims, keepdim: bool) -> Result<Tensor<T>, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError::new("Cannot compute max_dim of empty tensor"));
        }

        let dims_to_reduce = self.normalize_dims(&dims)?;
        let output_shape = self.compute_output_shape(&dims_to_reduce, keepdim);
        let output_size: usize = output_shape.iter().product();
        let mut output_data = vec![T::zero(); output_size];

        for (out_idx, out_val) in output_data.iter_mut().enumerate() {
            let base_indices = self.output_idx_to_indices(out_idx, &output_shape, &dims_to_reduce, keepdim);
            let first_val = self.get_first_in_reduced_dims(&base_indices, &dims_to_reduce)?;
            *out_val = self.reduce_over_dims(&base_indices, &dims_to_reduce,
                |acc, val| if val > acc { val } else { acc }, first_val)?;
        }

        Tensor::from_data(output_data, output_shape, self.device.clone())
    }

    /// Calculate the minimum value along specified dimensions.
    ///
    /// # Arguments
    /// * `dims` - Which dimensions to reduce (All, Single, or Multiple)
    /// * `keepdim` - Whether to keep reduced dimensions with size 1
    ///
    /// # Returns
    /// A new tensor with specified dimensions reduced
    ///
    /// # Errors
    /// When dimensions are out of bounds or tensor is empty
    fn min_dim(&self, dims: TensorDims, keepdim: bool) -> Result<Tensor<T>, TensorError> {
        if self.data.is_empty() {
            return Err(TensorError::new("Cannot compute min_dim of empty tensor"));
        }

        let dims_to_reduce = self.normalize_dims(&dims)?;
        let output_shape = self.compute_output_shape(&dims_to_reduce, keepdim);
        let output_size: usize = output_shape.iter().product();
        let mut output_data = vec![T::zero(); output_size];

        for (out_idx, out_val) in output_data.iter_mut().enumerate() {
            let base_indices = self.output_idx_to_indices(out_idx, &output_shape, &dims_to_reduce, keepdim);
            let first_val = self.get_first_in_reduced_dims(&base_indices, &dims_to_reduce)?;
            *out_val = self.reduce_over_dims(&base_indices, &dims_to_reduce,
                |acc, val| if val < acc { val } else { acc }, first_val)?;
        }

        Tensor::from_data(output_data, output_shape, self.device.clone())
    }
}

impl<T> Tensor<T>
where
    T: TensorNum,
{
    fn normalize_dims(&self, dims: &TensorDims) -> Result<Vec<usize>, TensorError> {
        match dims {
            TensorDims::All => Ok((0..self.shape.len()).collect()),
            TensorDims::Single(d) => {
                if *d >= self.shape.len() {
                    return Err(TensorError::new(&format!(
                        "Dimension {} is out of bounds for tensor with {} dimensions",
                        d, self.shape.len()
                    )));
                }
                Ok(vec![*d])
            }
            TensorDims::Multiple(dims_vec) => {
                for &d in dims_vec {
                    if d >= self.shape.len() {
                        return Err(TensorError::new(&format!(
                            "Dimension {} is out of bounds for tensor with {} dimensions",
                            d, self.shape.len()
                        )));
                    }
                }
                let mut sorted = dims_vec.clone();
                sorted.sort_unstable();
                sorted.dedup();
                Ok(sorted)
            }
        }
    }

    fn compute_output_shape(&self, dims_to_reduce: &[usize], keepdim: bool) -> Vec<usize> {
        let mut output_shape = Vec::new();
        for (i, &size) in self.shape.iter().enumerate() {
            if dims_to_reduce.contains(&i) {
                if keepdim {
                    output_shape.push(1);
                }
            } else {
                output_shape.push(size);
            }
        }
        if output_shape.is_empty() {
            output_shape.push(1);
        }
        output_shape
    }

    fn output_idx_to_indices(
        &self,
        out_idx: usize,
        output_shape: &[usize],
        dims_to_reduce: &[usize],
        keepdim: bool,
    ) -> Vec<usize> {
        let mut output_indices = Vec::with_capacity(output_shape.len());
        let mut remaining = out_idx;

        let output_strides = Tensor::<T>::calculate_strides(output_shape);
        for i in 0..output_shape.len() {
            output_indices.push(remaining / output_strides[i]);
            remaining %= output_strides[i];
        }

        let mut full_indices = vec![0; self.shape.len()];
        let mut out_pos = 0;

        for i in 0..self.shape.len() {
            if dims_to_reduce.contains(&i) {
                full_indices[i] = 0;
                if keepdim {
                    out_pos += 1;
                }
            } else {
                full_indices[i] = output_indices[out_pos];
                out_pos += 1;
            }
        }

        full_indices
    }

    fn reduce_over_dims<F>(
        &self,
        base_indices: &[usize],
        dims_to_reduce: &[usize],
        reduce_fn: F,
        init: T,
    ) -> Result<T, TensorError>
    where
        F: Fn(T, T) -> T,
    {
        let mut result = init;
        let reduce_sizes: Vec<usize> = dims_to_reduce.iter().map(|&d| self.shape[d]).collect();
        let total_iterations: usize = reduce_sizes.iter().product();

        for iter in 0..total_iterations {
            let mut indices = base_indices.to_vec();
            let mut remaining = iter;

            for (i, &dim) in dims_to_reduce.iter().enumerate().rev() {
                indices[dim] = remaining % reduce_sizes[i];
                remaining /= reduce_sizes[i];
            }

            let val = *self.get(&indices)?;
            result = reduce_fn(result, val);
        }

        Ok(result)
    }

    fn get_first_in_reduced_dims(
        &self,
        base_indices: &[usize],
        dims_to_reduce: &[usize],
    ) -> Result<T, TensorError> {
        let mut indices = base_indices.to_vec();
        for &dim in dims_to_reduce {
            indices[dim] = 0;
        }
        Ok(*self.get(&indices)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Device, Tensor, TensorCore, TensorDims, TensorStats, TensorStorage};

    #[test]
    fn test_sum() {
        let tensor = Tensor::<usize>::from_data(vec![1, 2, 3, 4], vec![2, 2], Device::CPU).unwrap();
        let sum = tensor.sum().unwrap();
        assert_eq!(sum, 10);
    }

    #[test]
    fn test_mean() {
        let tensor = Tensor::<usize>::from_data(vec![2, 4, 6, 8], vec![2, 2], Device::CPU).unwrap();
        let mean = tensor.mean().unwrap();
        assert_eq!(mean, 5);
    }

    #[test]
    fn test_min() {
        let tensor = Tensor::<usize>::from_data(vec![1, 2, 3, 4], vec![2, 2], Device::CPU).unwrap();
        let min = tensor.min().unwrap();
        assert_eq!(min, 1);
    }

    #[test]
    fn test_max() {
        let tensor = Tensor::<usize>::from_data(vec![1, 2, 3, 4], vec![2, 2], Device::CPU).unwrap();
        let max = tensor.max().unwrap();
        assert_eq!(max, 4);
    }

    #[test]
    fn test_stats_single_element() {
        let single_element = Tensor::<usize>::from_data(vec![42], vec![1], Device::CPU).unwrap();
        assert_eq!(single_element.sum().unwrap(), 42);
        assert_eq!(single_element.mean().unwrap(), 42);
        assert_eq!(single_element.min().unwrap(), 42);
        assert_eq!(single_element.max().unwrap(), 42);
    }

    #[test]
    fn test_sum_dim_2d_single_dim_no_keepdim() {
        let tensor = Tensor::<f32>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            Device::CPU
        ).unwrap();

        let result = tensor.sum_dim(TensorDims::Single(0), false).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(*result.get(&[0]).unwrap(), 5.0);
        assert_eq!(*result.get(&[1]).unwrap(), 7.0);
        assert_eq!(*result.get(&[2]).unwrap(), 9.0);

        let result = tensor.sum_dim(TensorDims::Single(1), false).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(*result.get(&[0]).unwrap(), 6.0);
        assert_eq!(*result.get(&[1]).unwrap(), 15.0);
    }

    #[test]
    fn test_sum_dim_2d_single_dim_keepdim() {
        let tensor = Tensor::<f32>::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            Device::CPU
        ).unwrap();

        let result = tensor.sum_dim(TensorDims::Single(0), true).unwrap();
        assert_eq!(result.shape(), &[1, 3]);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 5.0);
        assert_eq!(*result.get(&[0, 1]).unwrap(), 7.0);
        assert_eq!(*result.get(&[0, 2]).unwrap(), 9.0);

        let result = tensor.sum_dim(TensorDims::Single(1), true).unwrap();
        assert_eq!(result.shape(), &[2, 1]);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 6.0);
        assert_eq!(*result.get(&[1, 0]).unwrap(), 15.0);
    }

    #[test]
    fn test_sum_dim_3d_single_dim() {
        let tensor = Tensor::<f32>::from_data(
            (1..=24).map(|x| x as f32).collect(),
            vec![2, 3, 4],
            Device::CPU
        ).unwrap();

        let result = tensor.sum_dim(TensorDims::Single(1), false).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
        assert_eq!(*result.get(&[0, 0]).unwrap(), 15.0);
        assert_eq!(*result.get(&[0, 1]).unwrap(), 18.0);
        assert_eq!(*result.get(&[1, 0]).unwrap(), 51.0);
        assert_eq!(*result.get(&[1, 3]).unwrap(), 60.0);
    }

    #[test]
    fn test_sum_dim_all() {
        let tensor = Tensor::<f32>::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            Device::CPU
        ).unwrap();

        let result = tensor.sum_dim(TensorDims::All, false).unwrap();
        assert_eq!(result.shape(), &[1]);
        assert_eq!(*result.get(&[0]).unwrap(), 10.0);
    }

    #[test]
    fn test_sum_dim_multiple_dims() {
        let tensor = Tensor::<f32>::from_data(
            (1..=24).map(|x| x as f32).collect(),
            vec![2, 3, 4],
            Device::CPU
        ).unwrap();

        let result = tensor.sum_dim(TensorDims::Multiple(vec![0, 2]), false).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(*result.get(&[0]).unwrap(), 68.0);
        assert_eq!(*result.get(&[1]).unwrap(), 100.0);
        assert_eq!(*result.get(&[2]).unwrap(), 132.0);
    }

    #[test]
    fn test_mean_dim_2d() {
        let tensor = Tensor::<usize>::from_data(
            vec![2, 4, 6, 8, 10, 12],
            vec![2, 3],
            Device::CPU
        ).unwrap();

        let result = tensor.mean_dim(TensorDims::Single(1), false).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(*result.get(&[0]).unwrap(), 4);
        assert_eq!(*result.get(&[1]).unwrap(), 10);
    }

    #[test]
    fn test_max_dim_2d() {
        let tensor = Tensor::<f32>::from_data(
            vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0],
            vec![2, 3],
            Device::CPU
        ).unwrap();

        let result = tensor.max_dim(TensorDims::Single(1), false).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(*result.get(&[0]).unwrap(), 5.0);
        assert_eq!(*result.get(&[1]).unwrap(), 8.0);

        let result = tensor.max_dim(TensorDims::Single(0), false).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(*result.get(&[0]).unwrap(), 2.0);
        assert_eq!(*result.get(&[1]).unwrap(), 8.0);
        assert_eq!(*result.get(&[2]).unwrap(), 4.0);
    }

    #[test]
    fn test_min_dim_2d() {
        let tensor = Tensor::<f32>::from_data(
            vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0],
            vec![2, 3],
            Device::CPU
        ).unwrap();

        let result = tensor.min_dim(TensorDims::Single(1), false).unwrap();
        assert_eq!(result.shape(), &[2]);
        assert_eq!(*result.get(&[0]).unwrap(), 1.0);
        assert_eq!(*result.get(&[1]).unwrap(), 2.0);

        let result = tensor.min_dim(TensorDims::Single(0), false).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(*result.get(&[0]).unwrap(), 1.0);
        assert_eq!(*result.get(&[1]).unwrap(), 5.0);
        assert_eq!(*result.get(&[2]).unwrap(), 3.0);
    }

    #[test]
    fn test_max_min_dim_3d() {
        let tensor = Tensor::<f32>::from_data(
            (1..=24).map(|x| x as f32).collect(),
            vec![2, 3, 4],
            Device::CPU
        ).unwrap();

        let max_result = tensor.max_dim(TensorDims::Single(2), false).unwrap();
        assert_eq!(max_result.shape(), &[2, 3]);
        assert_eq!(*max_result.get(&[0, 0]).unwrap(), 4.0);
        assert_eq!(*max_result.get(&[1, 2]).unwrap(), 24.0);

        let min_result = tensor.min_dim(TensorDims::Single(2), false).unwrap();
        assert_eq!(min_result.shape(), &[2, 3]);
        assert_eq!(*min_result.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*min_result.get(&[1, 2]).unwrap(), 21.0);
    }

    #[test]
    fn test_dim_reduction_error_empty_tensor() {
        let result = Tensor::<f32>::from_data(vec![], vec![0], Device::CPU);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Tensor dimensions must be greater than 0"));
        }
    }

    #[test]
    fn test_dim_reduction_error_invalid_dim() {
        let tensor = Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::CPU).unwrap();

        let result = tensor.sum_dim(TensorDims::Single(5), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_mean_dim_error_invalid_dim() {
        let tensor = Tensor::<usize>::from_data(vec![1, 2, 3, 4], vec![2, 2], Device::CPU).unwrap();

        let result = tensor.mean_dim(TensorDims::Multiple(vec![0, 3]), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_keepdim_multiple_dims() {
        let tensor = Tensor::<f32>::from_data(
            (1..=24).map(|x| x as f32).collect(),
            vec![2, 3, 4],
            Device::CPU
        ).unwrap();

        let result = tensor.sum_dim(TensorDims::Multiple(vec![0, 2]), true).unwrap();
        assert_eq!(result.shape(), &[1, 3, 1]);
        assert_eq!(*result.get(&[0, 0, 0]).unwrap(), 68.0);
        assert_eq!(*result.get(&[0, 1, 0]).unwrap(), 100.0);
        assert_eq!(*result.get(&[0, 2, 0]).unwrap(), 132.0);
    }
}