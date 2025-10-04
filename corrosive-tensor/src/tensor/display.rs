use super::{Tensor, TensorCore, TensorFloat};
use std::fmt::{Display, Formatter, Result};

pub trait TensorDisplay<T>: Display {
    fn display_with_precision(&self, precision: usize) -> String;
    fn display_compact(&self) -> String;
    fn display_info(&self) -> String;
    fn display_shape(&self) -> String;
    fn display_formatted(&self, options: &DisplayOptions) -> String;
}

#[derive(Debug, Clone)]
pub struct DisplayOptions {
    pub precision: usize,
    pub max_elements: usize,
    pub compact: bool,
    pub show_shape: bool,
    pub show_dtype: bool,
    pub scientific_notation: bool,
}

impl Default for DisplayOptions {
    fn default() -> Self {
        DisplayOptions {
            precision: 4,
            max_elements: 1000,
            compact: false,
            show_shape: true,
            show_dtype: false,
            scientific_notation: false,
        }
    }
}

// TODO: Add support to display other types (integers, bools, etc.)
// TODO: Add support for displaying higher-dimensional tensors in a readable format
impl<T> Display for Tensor<T>
where
    T: TensorFloat,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.display_formatted(&DisplayOptions::default()))
    }
}

impl<T> TensorDisplay<T> for Tensor<T>
where
    T: TensorFloat,
{
    fn display_with_precision(&self, precision: usize) -> String {
        let options = DisplayOptions {
            precision,
            ..Default::default()
        };
        self.display_formatted(&options)
    }

    fn display_compact(&self) -> String {
        let options = DisplayOptions {
            compact: true,
            show_shape: false,
            ..Default::default()
        };
        self.display_formatted(&options)
    }

    fn display_info(&self) -> String {
        format!(
            "Tensor(shape={:?}, size={}, dtype={})",
            self.shape(),
            self.size(),
            std::any::type_name::<T>()
        )
    }

    fn display_shape(&self) -> String {
        format!("Shape: {:?}", self.shape())
    }

    fn display_formatted(&self, options: &DisplayOptions) -> String {
        let mut output = String::new();

        if options.show_shape {
            output.push_str(&format!("Tensor{:?}", self.shape()));
        } else {
            output.push_str("Tensor");
        }

        if options.show_dtype {
            output.push_str(&format!("<{}>", std::any::type_name::<T>()));
        }

        // Get CPU copy for display (transfers from CUDA if needed)
        let data = self.to_vec().expect("Failed to get tensor data for display");

        if options.compact && self.size() <= 10 {
            output.push('[');
            for (i, value) in data.iter().take(options.max_elements).enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                if options.scientific_notation {
                    output.push_str(&format!("{:.precision$e}", value, precision = options.precision));
                } else {
                    output.push_str(&format!("{:.precision$}", value, precision = options.precision));
                }
            }
            if self.size() > options.max_elements {
                output.push_str(", ...");
            }
            output.push(']');
        } else {
            output.push_str("[\n");

            if self.shape().len() == 1 {
                output.push_str("  [");
                for (i, value) in data.iter().take(options.max_elements).enumerate() {
                    if i > 0 {
                        output.push_str(", ");
                    }
                    if options.scientific_notation {
                        output.push_str(&format!("{:.precision$e}", value, precision = options.precision));
                    } else {
                        output.push_str(&format!("{:.precision$}", value, precision = options.precision));
                    }
                }
                if self.size() > options.max_elements {
                    output.push_str(", ...");
                }
                output.push(']');
            } else if self.shape().len() == 2 {
                let rows = self.shape()[0];
                let cols = self.shape()[1];
                let max_rows = options.max_elements / cols.max(1);

                for row in 0..rows.min(max_rows) {
                    output.push_str("  [");
                    for col in 0..cols {
                        if col > 0 {
                            output.push_str(", ");
                        }
                        let idx = row * cols + col;
                        if options.scientific_notation {
                            output.push_str(&format!("{:width$.precision$e}",
                                data[idx],
                                width = 8,
                                precision = options.precision
                            ));
                        } else {
                            output.push_str(&format!("{:width$.precision$}",
                                data[idx],
                                width = 8,
                                precision = options.precision
                            ));
                        }
                    }
                    output.push(']');
                    if row < rows - 1 {
                        output.push_str(",\n");
                    } else {
                        output.push('\n');
                    }
                }

                if rows > max_rows {
                    output.push_str("  ...\n");
                }
            } else {
                output.push_str("  [first ");
                output.push_str(&options.max_elements.min(self.size()).to_string());
                output.push_str(" elements: ");

                for (i, value) in data.iter().take(options.max_elements).enumerate() {
                    if i > 0 {
                        output.push_str(", ");
                    }
                    if options.scientific_notation {
                        output.push_str(&format!("{:.precision$e}", value, precision = options.precision));
                    } else {
                        output.push_str(&format!("{:.precision$}", value, precision = options.precision));
                    }
                }

                if self.size() > options.max_elements {
                    output.push_str(", ...");
                }
                output.push(']');
            }

            output.push_str("\n]");
        }

        output
    }
}
