use rand_distr::{Distribution, Uniform as RandUniform, Normal as RandNormal};
use rand::rngs::StdRng;
use rand::SeedableRng;
use num_traits::NumCast;
use corrosive_tensor::prelude::*;

#[derive(Debug, Clone)]
pub enum InitializerType {
    XavierUniform,
    XavierNormal,
    HeUniform,
    HeNormal,
    Normal { mean: f64, std: f64 },
    Uniform { low: f64, high: f64 },
}

pub struct Initializer {
    init_type: InitializerType,
    seed: Option<u64>,
}

impl Initializer {
    /// Creates a new Xavier Uniform initializer
    pub fn xavier_uniform() -> Self {
        Initializer {
            init_type: InitializerType::XavierUniform,
            seed: None,
        }
    }

    /// Creates a new Xavier Normal initializer
    pub fn xavier_normal() -> Self {
        Initializer {
            init_type: InitializerType::XavierNormal,
            seed: None,
        }
    }

    /// Creates a new He Uniform initializer
    pub fn he_uniform() -> Self {
        Initializer {
            init_type: InitializerType::HeUniform,
            seed: None,
        }
    }

    /// Creates a new He Normal initializer
    pub fn he_normal() -> Self {
        Initializer {
            init_type: InitializerType::HeNormal,
            seed: None,
        }
    }

    /// Creates a new Normal initializer
    pub fn normal(mean: f64, std: f64) -> Self {
        Initializer {
            init_type: InitializerType::Normal { mean, std },
            seed: None,
        }
    }

    /// Creates a new Uniform initializer
    pub fn uniform(low: f64, high: f64) -> Self {
        Initializer {
            init_type: InitializerType::Uniform { low, high },
            seed: None,
        }
    }

    /// Set a seed for reproducible initialization
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Initializes the matrix with the configured distribution
    ///
    /// # Arguments
    /// * `matrix` - The matrix to be initialized
    ///
    /// # Returns
    /// `Ok(())` on success
    ///
    /// # Errors
    /// Returns `MatrixError` when:
    /// - Matrix dimensions are incompatible with the initialization type
    /// - Distribution creation fails
    /// - Matrix filling fails
    pub fn initialize<T>(&self, shape: &[usize]) -> Result<Tensor<T>, TensorError>
    where
        T: TensorFloat,
    {
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut thread_rng = rand::rng();
                StdRng::from_rng(&mut thread_rng)
            },
        };

        // Generate values using the appropriate distribution
        let values = self.sample_values(
            shape, shape.iter().cloned().product(), &mut rng
        )?;

        let tensor = Tensor::<T>::from_data(values, shape.to_vec())?;
        Ok(tensor)
    }

    fn sample_values<T>(
        &self, shape: &[usize], count: usize, rng: &mut StdRng
    ) -> Result<Vec<T>, TensorError>
    where
        T: TensorFloat,
    {
        match &self.init_type {
            InitializerType::XavierUniform => {
                if shape.len() != 2 {
                    return Err(TensorError::new("Xavier uniform initialization requires a 2D matrix"));
                }
                let fan_in = shape[0];
                let fan_out = shape[1];
                let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();

                let uniform = RandUniform::new(-limit, limit)
                    .map_err(|e| TensorError::new(&format!("Failed to create uniform distribution: {}", e)))?;
                let values = (0..count)
                    .map(|_| NumCast::from(uniform.sample(rng))
                    .ok_or_else(|| TensorError::new("Failed to convert sample to target type")))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(values)
            },

            InitializerType::XavierNormal => {
                if shape.len() != 2 {
                    return Err(TensorError::new("Xavier normal initialization requires a 2D matrix"));
                }
                let fan_in = shape[0];
                let fan_out = shape[1];
                let std_dev = (2.0 / (fan_in + fan_out) as f64).sqrt();

                let normal = RandNormal::new(0.0, std_dev)
                    .map_err(|e| TensorError::new(&format!("Failed to create normal distribution: {}", e)))?;
                let values = (0..count)
                    .map(|_| NumCast::from(normal.sample(rng))
                    .ok_or_else(|| TensorError::new("Failed to convert sample to target type")))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(values)
            },

            InitializerType::HeUniform => {
                if shape.len() != 2 {
                    return Err(TensorError::new("He uniform initialization requires a 2D matrix"));
                }
                let fan_in = shape[0];
                let limit = (6.0 / fan_in as f64).sqrt();

                let uniform = RandUniform::new(-limit, limit)
                    .map_err(|e| TensorError::new(&format!("Failed to create uniform distribution: {}", e)))?;
                let values = (0..count)
                    .map(|_| NumCast::from(uniform.sample(rng))
                    .ok_or_else(|| TensorError::new("Failed to convert sample to target type")))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(values)
            },

            InitializerType::HeNormal => {
                if shape.len() != 2 {
                    return Err(TensorError::new("He normal initialization requires a 2D matrix"));
                }
                let fan_in = shape[0];
                let std_dev = (2.0 / fan_in as f64).sqrt();

                let normal = RandNormal::new(0.0, std_dev)
                    .map_err(|e| TensorError::new(&format!("Failed to create normal distribution: {}", e)))?;
                let values = (0..count)
                    .map(|_| NumCast::from(normal.sample(rng))
                    .ok_or_else(|| TensorError::new("Failed to convert sample to target type")))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(values)
            },

            InitializerType::Normal { mean, std } => {
                let normal = RandNormal::new(*mean, *std)
                    .map_err(|e| TensorError::new(&format!("Failed to create normal distribution: {}", e)))?;
                let values = (0..count)
                    .map(|_| NumCast::from(normal.sample(rng))
                    .ok_or_else(|| TensorError::new("Failed to convert sample to target type")))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(values)
            },

            InitializerType::Uniform { low, high } => {
                let uniform = RandUniform::new(*low, *high)
                    .map_err(|e| TensorError::new(&format!("Failed to create uniform distribution: {}", e)))?;
                let values = (0..count)
                    .map(|_| NumCast::from(uniform.sample(rng))
                    .ok_or_else(|| TensorError::new("Failed to convert sample to target type")))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(values)
            },
        }
    }
}
