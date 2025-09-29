pub enum TensorDims {
    All,
    Single(usize),
    Multiple(Vec<usize>)
}