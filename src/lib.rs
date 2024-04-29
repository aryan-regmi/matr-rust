use std::{marker::PhantomData, ops::Index};

/// A representation of a multi-dimensional vector.
struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.data
            .get(Self::vec_idx(index.0, index.1, self.cols))
            .expect("IndexOutOfBounds: The index is outside the matrix's bounds")
    }
}

impl Matrix {
    // TODO: Add push_row/push_col funcs

    /// Converts to the vector index from a matrix index.
    fn vec_idx(row: usize, col: usize, num_cols: usize) -> usize {
        num_cols * row + col
    }

    /// Creates a new, empty matrix with the specified size.
    pub fn new(rows: usize, cols: usize) -> Self {
        // Input validation
        {
            if rows == 0 || cols == 0 {
                panic!("InvalidSize: The matrix must have more than zero rows and columns")
            }
        }

        Self {
            data: Vec::with_capacity(rows * cols),
            rows,
            cols,
        }
    }

    /// Creates a matrix of the specified size with elements from the given vector.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    /// * Panics if the length of the vector is not equal to `num_cols * num_rows`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        // Input validation
        {
            if rows == 0 || cols == 0 {
                panic!("InvalidSize: The matrix must have more than zero rows and columns")
            }

            if data.len() != rows * cols {
                panic!(
                    "InvalidVector: The length of the vector must be {}",
                    rows * cols
                )
            }
        }

        Self { data, rows, cols }
    }

    /// Creates a new matrix of the specified size filled with the `default_value`.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn with_value(rows: usize, cols: usize, default_value: f32) -> Self {
        // Input validation
        {
            if rows == 0 || cols == 0 {
                panic!("InvalidSize: The matrix must have more than zero rows and columns")
            }
        }

        Self {
            data: vec![default_value; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates an identity matrix of the specified size.
    ///
    /// # Panics
    /// * Panics if the size is zero.
    pub fn identity(size: usize) -> Self {
        // Input validation
        {
            if size == 0 {
                panic!("InvalidSize: The matrix size must be greater than zero")
            }
        }

        let mut out = Self::from_vec(size, size, vec![0.0; size * size]);
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    out.data[Self::vec_idx(i, j, out.cols)] = 1.0;
                }
            }
        }
        out
    }

    /// Creates a matrix of the specified size, filled with `0`.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        // Input validation
        {
            if rows == 0 || cols == 0 {
                panic!("InvalidSize: The matrix must have more than zero rows and columns")
            }
        }

        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates a matrix of the specified size, filled with `1`.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn ones(rows: usize, cols: usize) -> Self {
        // Input validation
        {
            if rows == 0 || cols == 0 {
                panic!("InvalidSize: The matrix must have more than zero rows and columns")
            }
        }

        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Returns the specified row of the matrix.
    pub fn row<'a>(&'a self, idx: usize) -> Row<'a> {
        Row::new(self, idx)
    }

    /// Returns the specified column of the matrix.
    pub fn col<'a>(&'a self, idx: usize) -> Col<'a> {
        Col::new(self, idx)
    }
}

/// Represents a row of a matrix.
struct Row<'a> {
    mat: &'a Matrix,
    idx: usize,
    len: usize,
}

impl<'a> Row<'a> {
    fn new(mat: &'a Matrix, idx: usize) -> Self {
        Self {
            mat,
            idx,
            len: mat.cols,
        }
    }
}

/// Represents a column of a matrix.
struct Col<'a> {
    mat: &'a Matrix,
    idx: usize,
    len: usize,
}

impl<'a> Col<'a> {
    fn new(mat: &'a Matrix, idx: usize) -> Self {
        Col {
            mat,
            idx,
            len: mat.rows,
        }
    }
}
