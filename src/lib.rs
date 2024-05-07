use std::{
    alloc::{self, Layout},
    mem::{self, size_of},
    ops::{Index, IndexMut},
    ptr::{self, slice_from_raw_parts, slice_from_raw_parts_mut, NonNull},
};

/// Returns 1 if the value is positivie, 0 if it's zero, and -1 if it is negative.
fn sign(val: f32) -> f32 {
    if val == 0. {
        0.
    } else if val > 0. {
        1.
    } else {
        -1.
    }
}

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

    /// Returns a new matrix with the values from the specified row of the matrix.
    pub fn row(&self, idx: usize) -> Self {
        // Input validation
        if idx >= self.nrows {
            panic!(
                "Invalid row: The index must be less than {} (was {})",
                self.nrows, idx
            )
        }

        let mut out = Self::zeros(1, self.ncols);
        for j in 0..self.ncols {
            out[(0, j)] = self[(idx, j)];
        }

        out
    }

    /// Returns a new matrix with the values from the specified column of the matrix.
    pub fn col(&self, idx: usize) -> Self {
        // Input validation
        if idx >= self.ncols {
            panic!(
                "Invalid column: The index must be less than {} (was {})",
                self.ncols, idx
            )
        }

        let mut out = Self::zeros(self.nrows, 1);
        for i in 0..self.nrows {
            out[(i, 0)] = self[(i, idx)];
        }

        out
    }

    /// Sets the specified row to the given one.
    ///
    /// # Panics
    /// Panics if the row index (`idx`) is larger than the number of rows in the matrix.
    pub fn set_row(&mut self, idx: usize, row: &[f32]) {
        // Input validation
        {
            if idx >= self.nrows {
                panic!(
                    "Invalid row index: The row index must be less than {} (was {})",
                    self.nrows, idx
                )
            }

            if row.len() != self.ncols {
                panic!(
                    "Invalid row length: The row must have {} elements (has {})",
                    self.ncols,
                    row.len()
                )
            }
        }

        for j in 0..self.ncols {
            self[(idx, j)] = row[j];
        }
    }

    /// Sets the specified column to the given one.
    ///
    /// # Panics
    /// Panics if the column index (`idx`) is larger than the number of columns in the matrix.
    pub fn set_col(&mut self, idx: usize, col: &[f32]) {
        // Input validation
        {
            if idx >= self.ncols {
                panic!(
                    "Invalid column index: The column index must be less than {} (was {})",
                    self.ncols, idx
                )
            }

            if col.len() != self.nrows {
                panic!(
                    "Invalid column length: The column must have {} elements (has {})",
                    self.nrows,
                    col.len()
                )
            }
        }

        for i in 0..self.nrows {
            self[(i, idx)] = col[i];
        }
    }

    /// Increases the capacity of the matrix.
    fn resize(&mut self) {
        // Increase the capacity
        const RESIZE_FACTOR: usize = 2;
        let mut new_cap = self.capacity * RESIZE_FACTOR;
        if self.nrows == 0 {
            new_cap = self.ncols * RESIZE_FACTOR;
        } else if self.ncols == 0 {
            new_cap = self.nrows * RESIZE_FACTOR;
        }

        // Create new matrix w/ the new capacity and copy values into it
        let mut resized = Self::with_capacity(new_cap, 1);
        resized.nrows = self.nrows;
        resized.ncols = self.ncols;
        let resized_data = unsafe {
            slice_from_raw_parts_mut(resized.data.as_ptr(), self.nrows * self.ncols)
                .as_mut()
                .expect("Invalid slice")
        };
        let old_data = unsafe {
            slice_from_raw_parts(self.data.as_ptr(), self.nrows * self.ncols)
                .as_ref()
                .expect("Invalid slice")
        };
        resized_data.clone_from_slice(old_data);

        // Increase the capacity
        *self = resized;
    }

    /// Appends the row to the end of the matrix.
    pub fn push_row(&mut self, row: &[f32]) {
        if self.ncols == 0 {
            self.ncols = row.len();
        }

        // Resize if not enough space
        if self.capacity < (self.nrows + 1) * self.ncols {
            self.resize();
        }
        self.nrows += 1;

        self.set_row(self.nrows - 1, row);
    }

    /// Appends the column to the end of the matrix.
    pub fn push_col(&mut self, col: &[f32]) {
        if self.nrows == 0 {
            self.nrows = col.len();
        }

        // Resize if not enough space
        if self.capacity < (self.ncols + 1) * self.nrows {
            self.resize();
        }

        for i in 1..col.len() + 1 {
            unsafe {
                // Shift everything after the index to the right
                let idx = self.ncols * i + (i - 1);
                let nshifted = self.ncols * self.nrows - self.ncols * i;
                ptr::copy(
                    self.data.as_ptr().add(idx),
                    self.data.as_ptr().add(idx + 1),
                    nshifted,
                );

                // Update index with value from `col`
                *self.data.as_ptr().add(idx) = col[i - 1];
            }
        }

        self.ncols += 1;
    }

    // TODO: Add in-place versions of `transpose` and `scale`?

    /// Returns the transpose of the matrix.
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.ncols, self.nrows);
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// Calculates the Euclidean norm of the matrix.
    pub fn norm(&self) -> f32 {
        let mut norm = 0.0;
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                norm += self[(i, j)].powi(2)
            }
        }
        norm.sqrt()
    }

    /// Returns `true` if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Returns `true` if the matrix is an upper triangular.
    pub fn is_triangular_upper(&self) -> bool {
        if self.is_square() {
            for i in 1..self.nrows {
                for j in 0..i {
                    if self[(i, j)] != 0.0 {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Returns `true` if the matrix is an lower triangular.
    pub fn is_triangular_lower(&self) -> bool {
        if self.is_square() {
            for i in 0..self.nrows - 1 {
                for j in i + 1..self.nrows {
                    if self[(i, j)] != 0.0 {
                        return false;
                    }
                }
            }
            true
        } else {
            false
        }
    }

    /// Returns a new matrix with each element from `self` multiplied by the `scalar`.
    pub fn scale(&self, scalar: f32) -> Self {
        let mut out = self.clone();
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                out[(i, j)] *= scalar;
            }
        }
        out
    }

    /// Computes and returns the matrix multiplication of `self * other`.
    pub fn multiply(&self, other: &Self) -> Self {
        // Input validation
        if self.ncols != other.nrows {
            panic!(
                "Invalid size: The `other` matrix must have {} rows",
                self.ncols
            )
        }

        let mut out = Self::zeros(self.nrows, other.ncols);
        for i in 0..self.nrows {
            for j in 0..other.ncols {
                let mut sum = 0.0;
                for k in 0..self.ncols {
                    sum += self[(i, k)] * other[(k, j)];
                }
                out[(i, j)] = sum;
            }
        }
        out
    }

    /// Multiples each element of `self` with each corresponding element of `other`.
    pub fn multiply_elems(&self, other: &Self) -> Self {
        if self.size() != other.size() {
            panic!("Invalid size: The matricies must have the same size for element-wise multiplication")
        }

        let mut out = Self::zeros(self.nrows, self.ncols);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                out[(i, j)] = self[(i, j)] * other[(i, j)];
            }
        }
        out
    }

    /// Calculates the inverse of a lower triangular matrix (forward substitution).
    fn lower_triangular_inverse(&self) -> Option<Self> {
        let n = self.nrows;
        let mut out = Self::zeros(n, n);

        // Forward substitution
        for i in 0..n {
            // Check if matrix is singular
            if self[(i, i)] == 0.0 {
                return None;
            }

            out[(i, i)] = 1.0 / self[(i, i)];

            for j in 0..i {
                for k in j..i {
                    out[(i, j)] += self[(i, k)] * out[(k, j)];
                    out[(i, j)] = -out[(i, j)] / self[(i, i)];
                }
            }
        }

        Some(out)
    }

    /// Calculates the inverse of an upper triangular matrix (back substitution).
    fn upper_triangular_inverse(&self) -> Option<Self> {
        let n = self.nrows;
        let mut out = self.clone();

        // Back substitution
        for i in (1..=n - 1).rev() {
            // Check if matrix is singular
            if self[(i, i)] == 0.0 {
                return None;
            }

            out[(i, i)] = 1.0 / self[(i, i)];

            for j in (0..=i - 1).rev() {
                let mut sum = 0.0;
                for k in (j + 1..=i).rev() {
                    sum = sum - out[(j, k)] * out[(k, i)];
                }

                out[(j, i)] = sum / out[(j, j)];
            }
        }

        Some(out)
    }

    /// Performs LU decomposition of the matrix using Doolittle's method.
    ///
    /// Taken from [here](https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/).
    fn lu_decomposition(&self) -> (Self, Self) {
        let n = self.nrows;
        let mut lower = Matrix::zeros(n, n);
        let mut upper = Matrix::zeros(n, n);

        for i in 0..n {
            // Upper triangular
            for k in i..n {
                // Sum of `L[i,j] * U[j,k]`
                let mut sum = 0.0;
                for j in 0..i {
                    sum += lower[(i, j)] * upper[(j, k)];
                }

                // Evaluate `U[i,k]`
                upper[(i, k)] = self[(i, k)] - sum;
            }

            // Lower triangular
            for k in i..n {
                if i == k {
                    lower[(i, i)] = 1.0;
                } else {
                    // Sum of `L[k,j] * U[j,i]`
                    let mut sum = 0.0;
                    for j in 0..i {
                        sum += lower[(k, j)] * upper[(j, i)];
                    }

                    // Evaluate `L[k,i]`
                    lower[(k, i)] = (self[(k, i)] - sum) / upper[(i, i)];
                }
            }
        }

        (lower, upper)
    }

    /// Performs QR decomposition of the matrix using the Householder method.
    ///
    /// Taken from [here](https://kwokanthony.medium.com/detailed-explanation-with-example-on-qr-decomposition-by-householder-transformation-5e964d7f7656).
    fn qr_decomposition(&self) -> (Self, Self) {
        let (m, n) = self.size();
        let mut Qs: Vec<Self> = vec![];
        let mut Q = Self::identity(n);

        let mut curr = self;

        for k in 0..m {
            // Create sub matrix
            let mut sub_mat = Self::with_capacity(m - k, n - k);
            for i in 0..m {
                let mut row = vec![];
                for j in 0..n {
                    if i < k || j < k {
                        continue;
                    }
                    row.push(curr[(i, j)]);
                }
                if row.len() > 0 {
                    sub_mat.push_row(&row);
                }
            }

            // Calculate Q matrix
            {
                let sign = if sub_mat[(0, 0)] >= 0.0 { 1.0 } else { -1.0 };

                // Get the current column vector
                // let mut q =
            }
        }

        todo!()
    }

    /// Computes and the inverse of the matrix if one exists.
    pub fn inverse(&self) -> Option<Self> {
        if self.is_square() {
            if self.is_triangular_lower() {
                return self.lower_triangular_inverse();
            } else if self.is_triangular_upper() {
                return self.upper_triangular_inverse();
            } else {
                let (l, u) = self.lu_decomposition();
                return Some(u.inverse()?.multiply(&l.inverse()?));
            }
        } else {
            todo!("Implement QR Decomp")
        }
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

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            return false;
        }

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if self[(i, j)] != other[(i, j)] {
                    return false;
                }
            }
        }

        true
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        let mut out = Self::with_capacity(self.nrows, self.ncols);
        out.nrows = self.nrows;
        out.ncols = self.ncols;

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                out[(i, j)] = self[(i, j)];
            }
        }
        out
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

    #[test]
    fn can_set_rows() {
        let mut mat = Matrix::from_slice(2, 3, &[1., 2., 3., 4., 5., 6.]);
        mat.set_row(1, &[0.0; 3]);
        assert_eq!(mat, Matrix::from_slice(2, 3, &[1., 2., 3., 0., 0., 0.]))
    }

    #[test]
    fn can_set_cols() {
        let mut mat = Matrix::from_slice(2, 3, &[1., 2., 3., 4., 5., 6.]);
        mat.set_col(1, &[0.0; 2]);
        assert_eq!(mat, Matrix::from_slice(2, 3, &[1., 0., 3., 4., 0., 6.]))
    }

    #[test]
    fn can_push_rows() {
        let mut mat = Matrix::new();
        mat.push_row(&[1., 2., 3.]);
        mat.push_row(&[4., 5., 6.]);
        assert_eq!(mat, Matrix::from_slice(2, 3, &[1., 2., 3., 4., 5., 6.]));

        let mut mat = Matrix::with_capacity(5, 5);
        mat.push_row(&[1., 2., 3.]);
        mat.push_row(&[4., 5., 6.]);
        assert_eq!(mat, Matrix::from_slice(2, 3, &[1., 2., 3., 4., 5., 6.]));

        let mut mat = Matrix::from_slice(2, 3, &[1., 2., 3., 4., 5., 6.]);
        mat.push_row(&[7., 8., 9.]);
        assert_eq!(
            mat,
            Matrix::from_slice(3, 3, &[1., 2., 3., 4., 5., 6., 7., 8., 9.])
        );
    }

    #[test]
    fn can_push_cols() {
        let mut mat = Matrix::new();
        mat.push_col(&[1., 3.]);
        mat.push_col(&[2., 4.]);
        assert_eq!(mat, Matrix::from_slice(2, 2, &[1., 2., 3., 4.]));

        let mut mat = Matrix::from_slice(2, 2, &[1., 2., 3., 4.]);
        mat.push_col(&[5., 6.]);
        assert_eq!(mat, Matrix::from_slice(2, 3, &[1., 2., 5., 3., 4., 6.]));
    }

    #[test]
    fn can_scale() {
        let mat = Matrix::from_slice(2, 2, &[1., 2., 3., 4.]);
        assert_eq!(mat.scale(2.0), Matrix::from_slice(2, 2, &[2., 4., 6., 8.]));
    }

    #[test]
    fn can_transpose() {
        let mat = Matrix::from_slice(1, 4, &[1., 2., 3., 4.]);
        assert_eq!(mat.transpose(), Matrix::from_slice(4, 1, &[1., 2., 3., 4.]))
    }

    #[test]
    fn can_mul_matrix() {
        // Mat mul
        {
            let mat1 = Matrix::from_slice(2, 3, &[1., 2., 3., 4., 5., 6.]);
            let mat2 = Matrix::from_slice(3, 2, &[7., 8., 9., 10., 11., 12.]);
            let prod = mat1.multiply(&mat2);
            assert_eq!(prod, Matrix::from_slice(2, 2, &[58., 64., 139., 154.]));
        }

        // Elem mul
        {
            let mat1 = Matrix::from_slice(2, 2, &[1., 2., 3., 4.]);
            let mat2 = mat1.clone();
            let prod = mat1.multiply_elems(&mat2);
            assert_eq!(prod, Matrix::from_slice(2, 2, &[1., 4., 9., 16.]));
        }
    }

    #[test]
    fn can_check_triangular_matrix() {
        let mat = Matrix::from_slice(2, 2, &[1., 2., 0., 1.]);
        assert!(mat.is_square() == true);
        assert!(mat.is_triangular_lower() == false);
        assert!(mat.is_triangular_upper() == true);

        let mat = Matrix::from_slice(2, 2, &[1., 0., 2., 1.]);
        assert!(mat.is_square() == true);
        assert!(mat.is_triangular_lower() == true);
        assert!(mat.is_triangular_upper() == false);

        let mat = Matrix::from_slice(2, 3, &[1., 2., 3., 4., 5., 6.]);
        assert!(mat.is_square() == false);
        assert!(mat.is_triangular_lower() == false);
        assert!(mat.is_triangular_upper() == false);
    }

    #[test]
    fn can_invert_triangular_matrix() {
        // Lower triangular
        let mat = Matrix::from_slice(2, 2, &[1., 0., 2., 1.]);
        let inverse = mat.inverse().unwrap();
        assert_eq!(inverse, Matrix::from_slice(2, 2, &[1., 0., -2., 1.]));

        // Upper triangular
        let mat = Matrix::from_slice(2, 2, &[1., 2., 0., 1.]);
        let inverse = mat.inverse().unwrap();
        assert_eq!(inverse, Matrix::from_slice(2, 2, &[1., -2., 0., 1.]));
    }

    #[test]
    fn can_invert_square_matrix() {
        let mat = Matrix::from_slice(2, 2, &[1., 2., 3., 4.]);
        let inverse = mat.inverse().unwrap();
        assert_eq!(inverse, Matrix::from_slice(2, 2, &[-2., 1., 1.5, -0.5]));
    }
}
