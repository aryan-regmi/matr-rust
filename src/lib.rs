use std::{
    alloc::{alloc, Layout},
    mem,
    ops::{Index, IndexMut},
    ptr::{slice_from_raw_parts, slice_from_raw_parts_mut, NonNull},
};

struct Matrix {
    data: NonNull<f32>,
    nrows: usize,
    ncols: usize,
    capacity: usize,
}

impl Matrix {
    /// Converts to the linear index from a matrix index.
    pub fn vec_idx(row: usize, col: usize, num_cols: usize) -> usize {
        num_cols * row + col
    }

    /// Creates a new, empty matrix.
    pub fn new() -> Self {
        Self {
            data: NonNull::dangling(),
            nrows: 0,
            ncols: 0,
            capacity: 0,
        }
    }

    /// Creates a new, empty matrix with capacity for `nrows x ncols` elements.
    pub fn with_capacity(nrows: usize, ncols: usize) -> Self {
        let data = unsafe {
            let layout = Layout::from_size_align(nrows * ncols, 2).expect("Invalid memory layout"); // NOTE: Choose better align
            NonNull::new(mem::transmute(alloc(layout))).expect("Allocation failed")
        };
        Self {
            data,
            nrows: 0,
            ncols: 0,
            capacity: nrows * ncols,
        }
    }

    /// Creates a matrix of the specified size with elements from the given slice.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    /// * Panics if the length of the slice is not equal to `cols * rows`.
    pub fn from_slice(nrows: usize, ncols: usize, elems: &[f32]) -> Self {
        // Input validation
        {
            if nrows == 0 || ncols == 0 {
                panic!("Invalid size: The matrix must have more than zero rows and columns")
            }

            if elems.len() != nrows * ncols {
                panic!(
                    "Invalid slice: The length of the slice must be {}",
                    nrows * ncols
                )
            }
        }

        let mut out = Self::with_capacity(nrows, ncols);
        out.nrows = nrows;
        out.ncols = ncols;
        for i in 0..nrows {
            for j in 0..ncols {
                out[i][j] = elems[Self::vec_idx(i, j, ncols)];
            }
        }
        out
    }

    /// Creates a new matrix of the specified size filled with the `default_value`.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn with_value(nrows: usize, ncols: usize, default_value: f32) -> Self {
        Self::from_slice(nrows, ncols, &vec![default_value; nrows * ncols])
    }

    /// Creates an identity matrix of the specified size.
    ///
    /// # Panics
    /// * Panics if the size is zero.
    pub fn identity(size: usize) -> Self {
        let mut mat = Self::with_value(size, size, 0.0);
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    mat[i][j] = 1.0;
                }
            }
        }
        mat
    }

    /// Creates a matrix of the specified size, filled with `0`.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::with_value(rows, cols, 0.0)
    }

    /// Creates a matrix of the specified size, filled with `1`.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self::with_value(rows, cols, 1.0)
    }

    /// Returns the number of rows in the matrix.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns in the matrix.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the size of the matrix `(nrows, ncols)`.
    pub fn size(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns the number of elements allocated for the matrix.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Sets the specified row to the given row.
    pub fn set_row(&mut self, idx: usize, row: &[f32]) {
        self[idx].copy_from_slice(row);
    }

    /// Sets the specified column to the given column.
    pub fn set_col(&mut self, idx: usize, col: &[f32]) {
        for i in 0..self.nrows {
            self[i][idx] = col[i];
        }
    }
}

impl Index<usize> for Matrix {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        // Input validation
        if index >= self.nrows {
            panic!(
                "Invalid Index: The row index ({}) must be less than the number of rows ({})",
                index, self.nrows
            )
        }

        unsafe {
            slice_from_raw_parts(self.data.as_ptr().add(index * self.ncols), self.ncols)
                .as_ref()
                .expect("Invalid index")
        }
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // Input validation
        if index >= self.nrows {
            panic!(
                "Invalid Index: The row index ({}) must be less than the number of rows ({})",
                index, self.nrows
            )
        }

        unsafe {
            slice_from_raw_parts_mut(self.data.as_ptr().add(index * self.ncols), self.ncols)
                .as_mut()
                .expect("Invalid index")
        }
    }
}

impl std::fmt::Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Matrix ({} x {}) [", self.nrows, self.ncols))?;
        for i in 0..self.nrows {
            f.write_str("\n")?;
            for j in 0..self.ncols {
                f.write_fmt(format_args!(" {}", self[i][j]))?;
            }
        }
        f.write_str("\n]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_matrices() {
        let mat = Matrix::new();
        assert_eq!(mat.size(), (0, 0));
        assert_eq!(mat.capacity(), 0);

        let mat = Matrix::with_capacity(2, 3);
        assert_eq!(mat.size(), (0, 0));
        assert_eq!(mat.capacity(), 6);

        let elems = [1., 2., 3., 4., 5., 6.];
        let mat = Matrix::from_slice(2, 3, &elems);
        for i in 0..mat.nrows() {
            for j in 0..mat.ncols() {
                assert_eq!(mat[i][j], elems[Matrix::vec_idx(i, j, mat.ncols())])
            }
        }
        assert_eq!(mat.size(), (2, 3));
        assert_eq!(mat.capacity(), 6);
    }
}
