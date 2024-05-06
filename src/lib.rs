use std::{
    alloc::{self, Layout},
    ops::{Index, IndexMut},
    ptr::{slice_from_raw_parts, slice_from_raw_parts_mut, NonNull},
};

pub struct Matrix {
    data: NonNull<f32>,
    nrows: usize,
    ncols: usize,
    capacity: usize,
}

impl Matrix {
    /// Converts to the vector index from a matrix index.
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

    /// Creates a new, empty matrix with capacity for `nrows * ncols` elements.
    pub fn with_capacity(nrows: usize, ncols: usize) -> Self {
        if nrows == 0 || ncols == 0 {
            return Self::new();
        }

        let data = unsafe {
            let layout = Layout::array::<f32>(nrows * ncols).expect("Invalid memory layout");
            NonNull::new(std::mem::transmute(alloc::alloc(layout))).expect("Allocation failed")
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
    /// * Panics if the length of the slice is not equal to `nrows * ncols`.
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
                out[(i, j)] = elems[Self::vec_idx(i, j, ncols)];
            }
        }

        out
    }

    /// Creates a new matrix of the specified size filled with the `val`.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn with_value(nrows: usize, ncols: usize, val: f32) -> Self {
        // Input validation
        if nrows == 0 || ncols == 0 {
            panic!("Invalid size: The matrix must have more than zero rows and columns")
        }

        Self::from_slice(nrows, ncols, &vec![val; nrows * ncols])
    }

    /// Creates an identity matrix of the specified size.
    ///
    /// # Panics
    /// * Panics if the size is zero.
    pub fn identity(size: usize) -> Self {
        let mut out = Self::with_value(size, size, 0.0);
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    out[(i, j)] = 1.0;
                }
            }
        }
        out
    }

    /// Creates a matrix of the specified size, filled with `0`.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self::with_value(nrows, ncols, 0.0)
    }

    /// Creates a matrix of the specified size, filled with `1`.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn ones(nrows: usize, ncols: usize) -> Self {
        Self::with_value(nrows, ncols, 1.0)
    }

    /// Returns the size of the matrix: `(nrows, ncols)`.
    pub fn size(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns the number of rows in the matrix.
    pub fn num_rows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns in the matrix.
    pub fn num_cols(&self) -> usize {
        self.ncols
    }

    /// Returns an immutable reference to the specified row of the matrix.
    pub fn row(&self, idx: usize) -> &[f32] {
        // Input validation
        if idx >= self.nrows {
            panic!(
                "Invalid row: The index must be less than {} (was {})",
                self.nrows, idx
            )
        }

        unsafe {
            slice_from_raw_parts(self.data.as_ptr().add(idx * self.ncols), self.ncols)
                .as_ref()
                .expect("Invalid row index")
        }
    }

    /// Returns a mutable reference to the specified row of the matrix.
    pub fn row_mut(&mut self, idx: usize) -> &mut [f32] {
        // Input validation
        if idx >= self.nrows {
            panic!(
                "Invalid row: The index must be less than {} (was {})",
                self.nrows, idx
            )
        }

        unsafe {
            slice_from_raw_parts_mut(self.data.as_ptr().add(idx * self.ncols), self.ncols)
                .as_mut()
                .expect("Invalid row index")
        }
    }

    /// Returns an immutable reference to the specified column of the matrix.
    pub fn col(&self, idx: usize) -> Vec<&f32> {
        // Input validation
        if idx >= self.ncols {
            panic!(
                "Invalid column: The index must be less than {} (was {})",
                self.ncols, idx
            )
        }

        let mut out = vec![];
        for i in 0..self.nrows {
            out.push(&self[(i, idx)])
        }
        out
    }

    /// Returns a mutable reference to the specified column of the matrix.
    pub fn col_mut<'a>(&'a mut self, idx: usize) -> Vec<&'a mut f32> {
        // Input validation
        if idx >= self.ncols {
            panic!(
                "Invalid column: The index must be less than {} (was {})",
                self.ncols, idx
            )
        }

        let mut out = vec![];
        let nrows = self.nrows;
        let ncols = self.ncols;
        for i in 0..nrows {
            let val = unsafe {
                self.data
                    .as_ptr()
                    .add(Self::vec_idx(i, idx, ncols))
                    .as_mut()
                    .expect("Invalid index")
            };
            out.push(val)
        }
        out
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        // Input validation
        {
            let row = index.0;
            let col = index.1;
            if row >= self.nrows || col >= self.ncols {
                panic!(
                    "Invalid index: The row and col must be less than {} and {} respectively",
                    self.nrows, self.ncols
                )
            }
        }

        unsafe {
            self.data
                .as_ptr()
                .add(Self::vec_idx(index.0, index.1, self.ncols))
                .as_ref()
                .expect("Invalid index")
        }
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        // Input validation
        {
            let row = index.0;
            let col = index.1;
            if row >= self.nrows || col >= self.ncols {
                panic!(
                    "Invalid index: The row and col must be less than {} and {} respectively",
                    self.nrows, self.ncols
                )
            }
        }

        unsafe {
            self.data
                .as_ptr()
                .add(Self::vec_idx(index.0, index.1, self.ncols))
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
                f.write_fmt(format_args!(" {:?}", self[(i, j)]))?;
            }
        }
        f.write_str("\n]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn can_init_matrix() {
        let mat = Matrix::from_slice(2, 3, &[1., 2., 3., 4., 5., 6.]);
        dbg!(&mat);
        // dbg!(&mat.row(0));
        // dbg!(&mat.row(1));
        // mat.row_mut(0)[0] = 55.0;
        // dbg!(&mat);
        // dbg!(&mat.col(0));
        // dbg!(&mat.col(1));
        // dbg!(&mat.col(2));
        // *mat.col_mut(0)[1] = 55.0;
        // dbg!(&mat);

        let mat = Matrix::zeros(1, 3);
        dbg!(mat);

        let mat = Matrix::ones(2, 1);
        dbg!(mat);

        let mat = Matrix::identity(4);
        dbg!(mat);
    }
}
