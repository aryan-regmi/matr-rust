use std::{
    mem::ManuallyDrop,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
    ptr::NonNull,
};

// TODO: Better panic messages
//
// TODO: MatrixViews will do stuff inplace, Matrix will always create a new matrix!
//  - Enforce this and make views more efficient versions!
//
// TODO: Add func to solve Ax = b
//  - Add matrix decomp/determinant funcs
//
// TODO: Add inplace versions of algos!
//
// TODO: Add `to_vec` function
//
// TODO: Add function to reshape matrix.
//  - Add push/pop functions?
//  - Add funcs to access entire rows/cols
//
//  TODO: Add subslice struct/funcs

/// Returns a vector of size `n` with elements linearly spaced between `start` and `end`.
pub fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    let h = (end - start) / (n - 1) as f32;
    let mut x = vec![start; n];
    for i in 1..n {
        x[i] = x[i - 1] + h;
    }

    // Make sure vector elements are smaller than `end`
    if x[n - 1] > end {
        x[n - 1] = end;
    }

    x
}

/// Gets the sign of a number (behaves like `Matrix::sign`, rather than `signum`).
fn signum(val: f32) -> f32 {
    if val > 0.0 {
        1.0
    } else if val == 0.0 {
        0.0
    } else if val < 0.0 {
        -1.0
    } else {
        unreachable!()
    }
}

#[derive(Clone)]
pub struct MatrixView<'a> {
    matrix: &'a Matrix,
    start_row: usize,
    end_row: usize,
    start_col: usize,
    end_col: usize,
}

impl<'a> MatrixView<'a> {
    pub(crate) fn new(
        data: &'a Matrix,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> Self {
        Self {
            matrix: data,
            start_row: row_range.0,
            end_row: row_range.1,
            start_col: col_range.0,
            end_col: col_range.1,
        }
    }

    pub fn size(&self) -> (usize, usize) {
        let num_rows = (self.end_row - self.start_row) + 1;
        let num_cols = (self.end_col - self.start_col) + 1;

        (num_rows, num_cols)
    }

    pub fn num_rows(&self) -> usize {
        (self.end_row - self.start_row) + 1
    }

    pub fn num_cols(&self) -> usize {
        (self.end_col - self.start_col) + 1
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&f32> {
        // Input validation
        if row >= self.num_rows() || col >= self.num_cols() {
            return None;
        }

        self.matrix.get(self.start_row + row, self.start_col + col)
    }
}

impl<'a> Into<Matrix> for MatrixView<'a> {
    fn into(self) -> Matrix {
        self.matrix.clone()
    }
}

impl<'a> std::fmt::Debug for MatrixView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "MatrixView ({} x {}) \n[",
            self.num_rows(),
            self.num_cols()
        ))?;
        for i in 0..self.num_rows() {
            f.write_str("\t\n")?;
            for j in 0..self.num_cols() {
                f.write_fmt(format_args!(" {} ", self.get(i, j).expect("Invalid index")))?;
            }
        }
        f.write_str("\n]")
    }
}

impl<'a> Index<(usize, usize)> for MatrixView<'a> {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1).expect("Invalid index")
    }
}

pub struct MatrixViewMut<'a> {
    matrix: &'a mut Matrix,
    start_row: usize,
    end_row: usize,
    start_col: usize,
    end_col: usize,
}

impl<'a> MatrixViewMut<'a> {
    pub(crate) fn new(
        data: &'a mut Matrix,
        row_range: (usize, usize),
        col_range: (usize, usize),
    ) -> Self {
        Self {
            matrix: data,
            start_row: row_range.0,
            end_row: row_range.1,
            start_col: col_range.0,
            end_col: col_range.1,
        }
    }

    pub fn size(&self) -> (usize, usize) {
        let num_rows = (self.end_row - self.start_row) + 1;
        let num_cols = (self.end_col - self.start_col) + 1;

        (num_rows, num_cols)
    }

    pub fn num_rows(&self) -> usize {
        (self.end_row - self.start_row) + 1
    }

    pub fn num_cols(&self) -> usize {
        (self.end_col - self.start_col) + 1
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&f32> {
        // Input validation
        if row >= self.num_rows() || col >= self.num_cols() {
            return None;
        }

        self.matrix.get(self.start_row + row, self.start_col + col)
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f32> {
        // Input validation
        if row >= self.num_rows() || col >= self.num_cols() {
            return None;
        }

        self.matrix
            .get_mut(self.start_row + row, self.start_col + col)
    }

    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        // Input validation
        {
            if row >= self.num_rows() {
                panic!("The row index must be less than {}", self.num_rows())
            } else if col >= self.num_cols() {
                panic!("The column index must be less than {}", self.num_cols())
            }
        }

        let curr_value = self.get_mut(row, col).expect("Invalid index");
        *curr_value = value;
    }

    /// Multiply each element of the matrix by a scalar.
    pub fn scale(&mut self, scalar: f32) {
        for i in 0..self.num_rows() {
            for j in 0..self.num_cols() {
                let val = self.get_mut(i, j).expect("Invalid index");
                *val = *val * scalar;
            }
        }
    }

    /// Multiply the matrix with the `other` matrix.
    ///
    /// ## Panics
    /// * Panics if the dimensions of the matricies don't match.
    pub fn multiply(&self, other: &Self) -> Matrix {
        let mut out = Matrix::new(self.num_rows(), other.num_cols());
        for i in 0..self.num_rows() {
            for j in 0..other.num_cols() {
                let mut sum = 0.0;
                for k in 0..self.num_cols() {
                    sum = sum + self[(i, k)] * other[(k, j)];
                }
                out[(i, j)] = sum;
            }
        }
        out
    }
}

impl<'a> Into<Matrix> for MatrixViewMut<'a> {
    fn into(self) -> Matrix {
        self.matrix.clone()
    }
}

impl<'a> std::fmt::Debug for MatrixViewMut<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "MatrixView ({} x {}) \n[",
            self.num_rows(),
            self.num_cols()
        ))?;
        for i in 0..self.num_rows() {
            f.write_str("\t\n")?;
            for j in 0..self.num_cols() {
                f.write_fmt(format_args!(" {} ", self.get(i, j).expect("Invalid index")))?;
            }
        }
        f.write_str("\n]")
    }
}

impl<'a> Index<(usize, usize)> for MatrixViewMut<'a> {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1).expect("Invalid index")
    }
}

impl<'a> IndexMut<(usize, usize)> for MatrixViewMut<'a> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1).expect("Invalid index")
    }
}

/// A representation of a multi-dimensional vector.
pub struct Matrix {
    /// Number of rows.
    num_rows: usize,

    /// Number of columns.
    num_cols: usize,

    /// The memory array with the actual data.
    data: NonNull<f32>,
}

impl Matrix {
    /// Creates a new matrix with the specified size.
    ///
    /// The created matrix is initalized with zeros.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        // Input validation
        {
            if num_cols == 0 || num_rows == 0 {
                panic!("The number of rows and columns must be at least 1")
            }
        }

        let data = {
            let v = vec![0.0; num_rows * num_cols];
            let mut v = ManuallyDrop::new(v);
            NonNull::new(v.as_mut_ptr()).expect("Unable to allocate data")
        };

        Self {
            num_rows,
            num_cols,
            data,
        }
    }

    /// Creates a matrix of the specified size with elements from the given vector.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    /// * Panics if the length of the vector is not equal to `num_cols * num_rows`.
    pub fn from_vec(num_rows: usize, num_cols: usize, v: Vec<f32>) -> Self {
        // Input validation
        {
            if num_cols == 0 || num_rows == 0 {
                panic!("The number of rows and columns must be at least 1")
            }

            if v.len() != num_cols * num_rows {
                panic!("The vector must have {} elements", num_rows * num_cols)
            }
        }

        let data = {
            let v = v.clone();
            let mut v = ManuallyDrop::new(v);
            NonNull::new(v.as_mut_ptr()).expect("Unable to allocate data")
        };

        Self {
            num_rows,
            num_cols,
            data,
        }
    }

    /// Creates a new matrix with the specified size.
    ///
    /// The created matrix is initalized with the given value.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn with_default_value(num_rows: usize, num_cols: usize, value: f32) -> Self {
        // Input validation
        {
            if num_cols == 0 || num_rows == 0 {
                panic!("The number of rows and columns must be at least 1")
            }
        }

        let data = {
            let v = vec![value; num_rows * num_cols];
            let mut v = ManuallyDrop::new(v);
            NonNull::new(v.as_mut_ptr()).expect("Unable to allocate data")
        };

        Self {
            num_rows,
            num_cols,
            data,
        }
    }

    /// Creates an identity matrix of the specified size.
    pub fn identity(size: usize) -> Self {
        let mut out = Self::new(size, size);

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
    pub fn zeros(num_rows: usize, num_cols: usize) -> Self {
        Self::new(num_rows, num_cols)
    }

    /// Creates a matrix of the specified size, filled with `1`.
    pub fn ones(num_rows: usize, num_cols: usize) -> Self {
        Self::with_default_value(num_rows, num_cols, 1.0)
    }

    /// Converts from matrix indicies to vector index.
    fn idx(&self, row: usize, col: usize) -> usize {
        self.num_cols * row + col
    }

    /// Returns a reference to the matrix element at the specified row-column index.
    ///
    /// If the given indices are outside the matrix's bounds, `None` will be return.
    pub fn get(&self, row: usize, col: usize) -> Option<&f32> {
        // Input validation
        {
            if row >= self.num_rows || col >= self.num_cols {
                return None;
            }
        }

        let val = unsafe { self.data.as_ptr().add(self.idx(row, col)).as_ref() };

        val
    }

    /// Returns a mutable reference to the matrix element at the specified row-column index.
    ///
    /// If the given indices are outside the matrix's bounds, `None` will be return.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f32> {
        // Input validation
        {
            if row >= self.num_rows || col >= self.num_cols {
                return None;
            }
        }

        let val = unsafe { self.data.as_ptr().add(self.idx(row, col)).as_mut() };

        val
    }

    /// Sets the element at the specified matrix index to `value`.
    ///
    /// # Panics
    /// * Panics if the row or column indicies are outside the matrix's bounds.
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        // Input validation
        {
            if row >= self.num_rows {
                panic!("The row index must be less than {}", self.num_rows)
            } else if col >= self.num_cols {
                panic!("The column index must be less than {}", self.num_cols)
            }
        }

        let curr_value = self.get_mut(row, col).expect("Invalid index");
        *curr_value = value;
    }

    /// Creates a subslice/view of the matrix.
    pub fn view(&self, rows: (usize, usize), cols: (usize, usize)) -> MatrixView {
        // Input validation
        {
            if rows.0 > rows.1 {
                panic!("The starting row index in the subslice must be less than the ending row index ")
            } else if cols.0 > cols.1 {
                panic!("The starting column index in the subslice must be less than the ending column index ")
            }

            if rows.0 >= self.num_rows || rows.1 >= self.num_rows {
                panic!("IndexOutOfBounds: The start and end rows must be smaller than the number of rows in the matrix")
            } else if cols.0 >= self.num_cols || cols.1 >= self.num_cols {
                panic!("IndexOutOfBounds: The start and end columns must be smaller than the number of columns in the matrix")
            }
        }

        MatrixView::new(self, rows, cols)
    }

    /// Creates a mutable subslice/view of the matrix.
    pub fn view_mut(&mut self, rows: (usize, usize), cols: (usize, usize)) -> MatrixViewMut {
        // Input validation
        {
            if rows.0 > rows.1 {
                panic!("The starting row index in the subslice must be less than the ending row index ")
            } else if cols.0 > cols.1 {
                panic!("The starting column index in the subslice must be less than the ending column index ")
            }

            if rows.0 >= self.num_rows || rows.1 >= self.num_rows {
                panic!("IndexOutOfBounds: The start and end rows must be smaller than the number of rows in the matrix")
            } else if cols.0 >= self.num_cols || cols.1 >= self.num_cols {
                panic!("IndexOutOfBounds: The start and end columns must be smaller than the number of columns in the matrix")
            }
        }

        MatrixViewMut::new(self, rows, cols)
    }

    /// Multiply each element of the matrix by a scalar.
    pub fn scale(&mut self, scalar: f32) {
        for i in 0..self.num_rows * self.num_cols {
            let val = unsafe { self.data.as_ptr().add(i).as_mut().expect("Invalid index") };
            *val = *val * scalar;
        }
    }

    /// Multiply the matrix with the `other` matrix.
    ///
    /// ## Panics
    /// * Panics if the dimensions of the matricies don't match.
    pub fn multiply(&self, other: &Self) -> Self {
        // Input validation
        {
            if self.num_cols != other.num_rows {
                panic!(
                    "Invalid matrix dimensions: the other matrix must have {} rows",
                    self.num_cols
                )
            }
        }

        let mut out = Matrix::new(self.num_rows, other.num_cols);
        for i in 0..self.num_rows {
            for j in 0..other.num_cols {
                let mut sum = 0.0;
                for k in 0..self.num_cols {
                    sum = sum + self.get(i, k).unwrap() * other.get(k, j).unwrap();
                }
                out.set(i, j, sum)
            }
        }

        out
    }

    /// Returns the size of the matrix as `(rows, columns)`.
    pub const fn size(&self) -> (usize, usize) {
        (self.num_rows, self.num_cols)
    }

    /// Returns then number of rows in the matrix.
    pub const fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Returns then number of columns in the matrix.
    pub const fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Returns the transpose of the matrix.
    pub fn transpose(&self) -> Self {
        let mut transposed = Self::new(self.num_cols, self.num_rows);

        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                transposed.set(j, i, *self.get(i, j).expect("Invaid Index"));
            }
        }

        transposed
    }

    /// Returns `true` if the matrix is square.
    pub fn is_square(&self) -> bool {
        if self.num_cols == self.num_rows {
            true
        } else {
            false
        }
    }

    /// Returns `true` if the matrix is an upper triangular.
    pub fn is_triangular_upper(&self) -> bool {
        if self.is_square() {
            for i in 1..self.num_rows {
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
            for i in 0..self.num_rows - 1 {
                for j in i + 1..self.num_rows {
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

    /// Calculates the inverse of a lower triangular matrix (forward substitution).
    fn lower_triangular_inverse(&self) -> Option<Self> {
        let n = self.num_rows;
        let mut out = Matrix::zeros(n, n);

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
        let n = self.num_rows;
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
        let n = self.num_rows;
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

    /// Calculates the Euclidean norm of the matrix.
    pub fn norm(&self) -> f32 {
        let mut norm = 0.0;
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                norm += self[(i, j)].powi(2);
            }
        }
        return norm.sqrt();
    }

    /// Returns a matrix with the same size as `self` where each element is:
    ///     * `1` if the corresponding element is greater than `0`.
    ///     * `0` if the corresponding element is `0`.
    ///     * `-1` if the corresponding element is less than `0`.
    pub fn sign(&self) -> Self {
        let mut out = Matrix::zeros(self.num_rows, self.num_cols);

        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                let val = self[(i, j)];
                if val > 0.0 {
                    out[(i, j)] = 1.0;
                } else if val == 0.0 {
                    out[(i, j)] = 0.0;
                } else if val < 0.0 {
                    out[(i, j)] = -1.0;
                }
            }
        }

        out
    }

    /// Performs QR decomposition of the matrix using the Householder method.
    fn qr_decomposition(&self) -> (Self, Self) {
        let (m, n) = self.size();
        let mut q = Matrix::identity(m);
        let mut r = self.clone();

        for j in 0..n {
            // H = I-tau*w*w'
            let r_view_1: Matrix = r.view((j, m - 1), (j, j)).into();
            let normx = r_view_1.norm();
            let s = -signum(r[(j, j)]);
            let u1 = r[(j, j)] - s * normx;
            let mut w = r_view_1 / u1;
            w[(1, 1)] = 1.0;
            let tau = -s * u1 / normx;

            // R := HR, Q := QH
            let mut r_view_2 = r.view((j, m - 1), (0, n - 1));
            let mut q_view_1 = q.view((0, m - 1), (j, m - 1));
            for i in j..m {
                // Solve for R
                for k in 0..n {
                    // r[(i, k)] = r[(i, k)] - (tau * w) * (w.transpose() * r[(i, k)]);
                }

                // Solve for Q
                todo!()
            }
        }

        todo!()
    }

    /// Returns the inverse of the matrix.
    pub fn inverse(&self) -> Option<Self> {
        if self.is_square() {
            if self.is_triangular_upper() {
                return self.upper_triangular_inverse();
            } else if self.is_triangular_lower() {
                return self.lower_triangular_inverse();
            } else {
                // NOTE: Dont' invert triangulars for solution?
                //  - Check if there is more efficient ways

                let (l, u) = self.lu_decomposition();
                return Some(
                    u.inverse().expect("Invalid upper matrix")
                        * l.inverse().expect("Invalid lower matrix"),
                );
            }
        } else {
        }

        todo!()
    }
}

impl std::fmt::Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Matrix ({} x {}) \n[",
            self.num_rows, self.num_cols
        ))?;
        for i in 0..self.num_rows {
            f.write_str("\t\n")?;
            for j in 0..self.num_cols {
                f.write_fmt(format_args!(" {} ", self.get(i, j).expect("Invalid index")))?;
            }
        }
        f.write_str("\n]")
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        unsafe {
            Vec::from_raw_parts(
                self.data.as_ptr(),
                self.num_rows * self.num_cols,
                self.num_rows * self.num_cols,
            )
        };
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        let mut out = Matrix::zeros(self.num_rows, self.num_cols);

        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                out[(i, j)] = self[(i, j)];
            }
        }

        out
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1).expect("Invalid index")
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.get_mut(index.0, index.1).expect("Invalid index")
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if (self.num_rows != other.num_rows) || (self.num_cols != other.num_cols) {
            false
        } else {
            for i in 0..self.num_rows {
                for j in 0..self.num_cols {
                    if self[(i, j)] != other[(i, j)] {
                        return false;
                    }
                }
            }
            true
        }
    }
}

impl Add<Matrix> for Matrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Input validation
        if self.size() != rhs.size() {
            panic!("InvalidShape: The matricies must have the same shape/size")
        }

        let mut out = Matrix::zeros(self.num_rows, self.num_cols);
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                out[(i, j)] = self[(i, j)] + rhs[(i, j)];
            }
        }

        out
    }
}

impl Sub<f32> for Matrix {
    type Output = Self;

    fn sub(mut self, rhs: f32) -> Self::Output {
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                self[(i, j)] -= rhs;
            }
        }
        self
    }
}

impl Sub<Matrix> for f32 {
    type Output = Matrix;

    fn sub(self, mut rhs: Matrix) -> Self::Output {
        for i in 0..rhs.num_rows {
            for j in 0..rhs.num_cols {
                rhs[(i, j)] = self - rhs[(i, j)];
            }
        }
        rhs
    }
}

impl Sub<Matrix> for Matrix {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        // Input validation
        if self.size() != rhs.size() {
            panic!("InvalidShape: The matricies must have the same shape/size")
        }

        let mut out = Matrix::zeros(self.num_rows, self.num_cols);
        for i in 0..self.num_rows {
            for j in 0..self.num_cols {
                out[(i, j)] = self[(i, j)] - rhs[(i, j)];
            }
        }

        out
    }
}

impl Mul<f32> for Matrix {
    type Output = Self;

    fn mul(mut self, rhs: f32) -> Self::Output {
        self.scale(rhs);
        self
    }
}

impl Mul<f32> for &mut Matrix {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs);
        self
    }
}

impl Mul<Matrix> for f32 {
    type Output = Matrix;

    fn mul(self, mut rhs: Matrix) -> Self::Output {
        rhs.scale(self);
        rhs
    }
}

impl Div<f32> for Matrix {
    type Output = Self;

    fn div(mut self, rhs: f32) -> Self::Output {
        self.scale(1. / rhs);
        self
    }
}

impl Div<f32> for &mut Matrix {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self.scale(1. / rhs);
        self
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Self;

    fn mul(self, rhs: Matrix) -> Self::Output {
        self.multiply(&rhs)
    }
}

impl Mul<Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Self::Output {
        self.multiply(&rhs)
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        self.multiply(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_matrix() {
        let mat = Matrix::new(2, 3);
        assert_eq!(mat.num_rows(), 2);
        assert_eq!(mat.num_cols(), 3);

        let mat = Matrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(mat.size(), (2, 3));

        let mat = Matrix::identity(2);
        assert_eq!(mat.size(), (2, 2));
        assert_eq!(mat[(0, 0)], 1.0);
        assert_eq!(mat[(1, 1)], 1.0);
    }

    #[test]
    fn can_scale_matrix() {
        let mut mat = Matrix::new(2, 3);
        for i in 0..2 {
            for j in 0..3 {
                mat.set(i, j, 2.0);
            }
        }
        mat.scale(4.0);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(mat.get(i, j), Some(&8.0));
            }
        }
    }

    #[test]
    fn can_mul_matrix() {
        let mat1 = Matrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let mat2 = Matrix::from_vec(3, 2, vec![7., 8., 9., 10., 11., 12.]);

        let mult = mat1.multiply(&mat2);

        assert_eq!(mult[(0, 0)], 58.0);
        assert_eq!(mult[(0, 1)], 64.0);
        assert_eq!(mult[(1, 0)], 139.0);
        assert_eq!(mult[(1, 1)], 154.0);
    }

    #[test]
    fn can_transpose_matrix() {
        let mat1 = Matrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        assert_eq!(
            mat1.transpose(),
            Matrix::from_vec(3, 2, vec![1., 4., 2., 5., 3., 6.])
        )
    }

    #[test]
    fn can_check_triangular_matrix() {
        let mat = Matrix::from_vec(2, 2, vec![1., 2., 0., 1.]);
        assert!(mat.is_square() == true);
        assert!(mat.is_triangular_lower() == false);
        assert!(mat.is_triangular_upper() == true);

        let mat = Matrix::from_vec(2, 2, vec![1., 0., 2., 1.]);
        assert!(mat.is_square() == true);
        assert!(mat.is_triangular_lower() == true);
        assert!(mat.is_triangular_upper() == false);

        let mat = Matrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        assert!(mat.is_square() == false);
        assert!(mat.is_triangular_lower() == false);
        assert!(mat.is_triangular_upper() == false);
    }

    #[test]
    fn can_invert_lower_triangular_matrix() {
        let mat = Matrix::from_vec(2, 2, vec![1., 0., 2., 1.])
            .lower_triangular_inverse()
            .unwrap();

        assert_eq!(mat, Matrix::from_vec(2, 2, vec![1., 0., -2., 1.]));
    }

    #[test]
    fn can_invert_upper_triangular_matrix() {
        let mat = Matrix::from_vec(2, 2, vec![1., 2., 0., 1.])
            .upper_triangular_inverse()
            .unwrap();

        assert_eq!(mat, Matrix::from_vec(2, 2, vec![1., -2., 0., 1.]));
    }

    #[test]
    fn can_lu_decompose_matrix() {
        let mat = Matrix::from_vec(3, 3, vec![2., 7., 1., 3., -2., 0., 1., 5., 3.]);
        let (l, u) = mat.lu_decomposition();
        let lu = l.clone() * u.clone();

        assert_eq!(
            l,
            Matrix::from_vec(3, 3, vec![1., 0., 0., 1.5, 1., 0., 0.5, -0.12, 1.])
        );

        assert_eq!(
            u,
            Matrix::from_vec(3, 3, vec![2., 7., 1., 0., -12.5, -1.5, 0., 0., 2.32])
        );

        assert_eq!(
            lu,
            Matrix::from_vec(3, 3, vec![2., 7., 1., 3., -2., 0., 1., 5., 3.])
        );
    }

    #[test]
    fn can_norm_matrix() {
        let mat = Matrix::from_vec(2, 2, vec![1., -7., -2., -3.]);
        assert_eq!(mat.norm(), (63_f32).sqrt())
    }

    #[test]
    fn can_create_matrix_view() {
        let mat = Matrix::from_vec(2, 2, vec![1., -7., -2., -3.]);

        let view = mat.view((0, 1), (1, 1));
        assert_eq!(view[(0, 0)], mat[(0, 1)]);
    }
}
