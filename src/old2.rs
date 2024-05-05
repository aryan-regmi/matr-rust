use std::ops::{Div, Index, IndexMut, Mul};

// TODO: Make it work with all floats!
//
// TODO: Add Div, Add, Sub operators!
//  - Should happen per element (don't use `solve` method for `Div`)

/// Returns a matrix of size `1 x n` (row vector) with elements linearly spaced between `start` and `end`.
pub fn linspace(start: f32, end: f32, n: usize) -> Row {
    let h = (end - start) / (n - 1) as f32;
    let mut x = vec![start; n];
    for i in 1..n {
        x[i] = x[i - 1] + h;
    }

    // Make sure vector elements are smaller than `end`
    if x[n - 1] > end {
        x[n - 1] = end;
    }
    Row { data: x }
}

/// Represents a row of a matrix.
#[derive(Clone, PartialEq)]
pub struct Row {
    data: Vec<f32>,
}

impl Row {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Multiply each element of the row by a scalar.
    pub fn scale(&self, scalar: f32) -> Self {
        let mut scaled = vec![];
        for val in &self.data {
            scaled.push(*val * scalar);
        }
        Self { data: scaled }
    }

    /// Calculates the dot prodcut of `self` and `other`.
    pub fn dot(&self, other: &Self) -> f32 {
        // Input validation
        if self.data.len() != other.data.len() {
            panic!(
                "InvalidShape: `other` must have the same length as `self` ({})",
                self.data.len()
            )
        }

        let mut sum = 0.0;
        for i in 0..self.data.len() {
            sum += self[i] * other[i];
        }
        sum
    }

    /// Element wise multiplication of `self` and `other`.
    pub fn mul_elem(&self, other: &Row) -> Self {
        // Input validation
        if self.data.len() != other.data.len() {
            panic!(
                "InvalidShape: `other` must have the same length as `self` ({})",
                self.data.len()
            )
        }

        let mut prod = vec![];
        for i in 0..self.data.len() {
            prod.push(self[i] * other[i]);
        }
        Row::new(prod)
    }

    /// Multiplies `self` with the given column.
    fn mul_col(&self, other: &Col) -> f32 {
        // Input validation
        if self.data.len() != other.data.len() {
            panic!(
                "InvalidShape: `other` must have the same length as `self` ({})",
                self.data.len()
            )
        }

        let mut sum = 0.0;
        for i in 0..self.data.len() {
            sum += self[i] * other[i];
        }
        sum
    }

    /// Multiplies `self` with the given matrix.
    fn mul_mat(&self, other: &Matrix) -> Self {
        // Input validation
        if self.data.len() != other.rows {
            panic!(
                "InvalidShape: `other` must have the same number of rows as `self` ({})",
                self.data.len()
            )
        }

        let mut data = vec![];
        for j in 0..other.cols {
            let mut sum = 0.0;
            for i in 0..other.rows {
                sum += self[i] * other[i][j];
            }
            data.push(sum);
        }
        Self { data }
    }

    /// Divides the scalar by each element in the row.
    fn div_scalar(&self, scalar: f32) -> Self {
        let mut scaled = vec![];
        for elem in &self.data {
            scaled.push(scalar / (*elem));
        }
        Self { data: scaled }
    }
}

impl Index<usize> for Row {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Row {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl std::fmt::Debug for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[ ")?;
        for val in &self.data {
            f.write_fmt(format_args!(" {:?} ", val))?;
        }
        f.write_str(" ]")
    }
}

impl Mul<f32> for Row {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

impl Mul<f32> for &Row {
    type Output = Row;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

impl Mul<Row> for f32 {
    type Output = Row;

    fn mul(self, rhs: Row) -> Self::Output {
        rhs.scale(self)
    }
}

impl Mul<&Row> for f32 {
    type Output = Row;

    fn mul(self, rhs: &Row) -> Self::Output {
        rhs.scale(self)
    }
}

impl Mul<Col> for Row {
    type Output = f32;

    fn mul(self, rhs: Col) -> Self::Output {
        self.mul_col(&rhs)
    }
}

impl Mul<&Col> for &Row {
    type Output = f32;

    fn mul(self, rhs: &Col) -> Self::Output {
        self.mul_col(&rhs)
    }
}

impl Mul<Matrix> for Row {
    type Output = Self;

    fn mul(self, rhs: Matrix) -> Self::Output {
        self.mul_mat(&rhs)
    }
}

impl Mul<&Matrix> for &Row {
    type Output = Row;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        self.mul_mat(&rhs)
    }
}

impl Div<f32> for Row {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self.scale(1. / rhs)
    }
}

impl Div<f32> for &Row {
    type Output = Row;

    fn div(self, rhs: f32) -> Self::Output {
        self.scale(1. / rhs)
    }
}

impl Div<Row> for f32 {
    type Output = Row;

    fn div(self, rhs: Row) -> Self::Output {
        rhs.div_scalar(self)
    }
}

impl Div<&Row> for f32 {
    type Output = Row;

    fn div(self, rhs: &Row) -> Self::Output {
        rhs.div_scalar(self)
    }
}

impl Div<Row> for Row {
    type Output = f32;

    fn div(self, rhs: Row) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl Div<&Row> for &Row {
    type Output = f32;

    fn div(self, rhs: &Row) -> Self::Output {
        self * &(1.0 / rhs)
    }
}

// TODO: Add rest of Div

/// Represents a column of a matrix.
#[derive(Clone, PartialEq)]
pub struct Col {
    data: Vec<f32>,
}

impl Col {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Multiply each element of the row by a scalar.
    pub fn scale(&self, scalar: f32) -> Self {
        let mut scaled = vec![];
        for val in &self.data {
            scaled.push(*val * scalar);
        }
        Self { data: scaled }
    }

    // /// Calculates the dot prodcut of `self` and `other`.
    pub fn dot(&self, other: &Self) -> f32 {
        // Input validation
        if self.data.len() != other.data.len() {
            panic!(
                "InvalidShape: `other` must have the same length as `self` ({})",
                self.data.len()
            )
        }

        let mut sum = 0.0;
        for i in 0..self.data.len() {
            sum += self[i] * other[i];
        }
        sum
    }

    /// Multiplies `self` with the given row.
    pub fn mul_row(&self, other: &Row) -> Matrix {
        let m = self.data.len();
        let n = other.data.len();

        let mut rows = vec![];
        for i in 0..m {
            let mut row = Row::new(vec![]);
            for j in 0..n {
                row.data.push(self[i] * other[j]);
            }
            rows.push(row);
        }

        Matrix::from_rows(rows)
    }

    /// Divides the scalar by each element in the column.
    pub fn div_scalar(&self, scalar: f32) -> Self {
        let mut scaled = vec![];
        for elem in &self.data {
            scaled.push(scalar / (*elem));
        }
        Self { data: scaled }
    }
}

impl Index<usize> for Col {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Col {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl std::fmt::Debug for Col {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[\n")?;
        for val in &self.data {
            f.write_fmt(format_args!(" {:?}\n", val))?;
        }
        f.write_str("]")
    }
}

impl Mul<f32> for Col {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

impl Mul<Col> for f32 {
    type Output = Col;

    fn mul(self, rhs: Col) -> Self::Output {
        rhs.scale(self)
    }
}

impl Mul<f32> for &mut Col {
    type Output = Col;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

impl Mul<Row> for Col {
    type Output = Matrix;

    fn mul(self, rhs: Row) -> Self::Output {
        self.mul_row(&rhs)
    }
}

impl Mul<&Row> for &Col {
    type Output = Matrix;

    fn mul(self, rhs: &Row) -> Self::Output {
        self.mul_row(rhs)
    }
}

impl Mul<Col> for Col {
    type Output = f32;
    fn mul(self, rhs: Col) -> Self::Output {
        self.dot(&rhs)
    }
}

impl Mul<&Col> for &Col {
    type Output = f32;
    fn mul(self, rhs: &Col) -> Self::Output {
        self.dot(rhs)
    }
}

impl Div<f32> for Col {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self.scale(1. / rhs)
    }
}

impl Div<f32> for &Col {
    type Output = Col;

    fn div(self, rhs: f32) -> Self::Output {
        self.scale(1. / rhs)
    }
}

impl Div<Col> for f32 {
    type Output = Col;

    fn div(self, rhs: Col) -> Self::Output {
        rhs.div_scalar(self)
    }
}

impl Div<&Col> for f32 {
    type Output = Col;

    fn div(self, rhs: &Col) -> Self::Output {
        rhs.div_scalar(self)
    }
}

/// Represents a multi-dimensional matrix.
///
/// # Safety
/// When mutably accessing rows of the matrix, care must be taken to ensure that the length of any
/// modified row equals the number of rows in the matrix; otherwise, an invalid matrix will be
/// created:
///
/// ```rust
/// let mut mat = Matrix::from_slice(2,2 &[1., 2., 3., 4.]);
/// mat[1] = Row::new(vec![0., 0., 0.]); // More columns in this row than the matrix
/// ```
#[derive(Clone, PartialEq)]
pub struct Matrix {
    data: Vec<Row>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Converts to the vector index from a matrix index.
    fn vec_idx(row: usize, col: usize, num_cols: usize) -> usize {
        num_cols * row + col
    }

    /// Creates a new, empty matrix.
    pub fn new() -> Self {
        Self {
            data: vec![],
            rows: 0,
            cols: 0,
        }
    }

    /// Creates a new, empty matrix with capacity for the specified number of rows.
    ///
    /// # Note
    /// It is currently impossible to specify the capacity for columns.
    pub fn with_capacity(rows: usize) -> Self {
        let data = Vec::with_capacity(rows);
        Self {
            data,
            rows: 0,
            cols: 0,
        }
    }

    /// Creates a matrix from the vector of rows.
    ///
    /// # Panics
    /// * Panics if all the rows' lengths are not equal (i.e number of columns must be the same).
    pub fn from_rows(row_vec: Vec<Row>) -> Self {
        let m = row_vec.len();
        let n = row_vec[0].data.len();

        let mut rows = vec![];
        for (i, row) in row_vec.into_iter().enumerate() {
            // Check that all rows have the same number of elements
            if row.data.len() != n {
                panic!("InvalidRows: All rows must have the same number of elements ({}), but row {} in the given vector has {} elements", n, i,row.data.len())
            }
            rows.push(row);
        }

        Self {
            data: rows,
            rows: m,
            cols: n,
        }
    }

    /// Creates a matrix of the specified size with elements from the given slice.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    /// * Panics if the length of the slice is not equal to `cols * rows`.
    pub fn from_slice(rows: usize, cols: usize, v: &[f32]) -> Self {
        // Input validation
        {
            if rows == 0 || cols == 0 {
                panic!("InvalidSize: The matrix must have more than zero rows and columns")
            }

            if v.len() != rows * cols {
                panic!(
                    "InvalidVector: The length of the vector must be {}",
                    rows * cols
                )
            }
        }

        let mut data = vec![];
        for i in 0..rows {
            let mut row = Row::new(vec![]);
            for j in 0..cols {
                row.data.push(v[Self::vec_idx(i, j, cols)]);
            }
            data.push(row);
        }
        Self { data, rows, cols }
    }

    /// Creates a new matrix of the specified size filled with the `default_value`.
    ///
    /// # Panics
    /// * Panics if the number of rows or columns is zero.
    pub fn with_value(rows: usize, cols: usize, default_value: f32) -> Self {
        Self::from_slice(rows, cols, &vec![default_value; rows * cols])
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

    /// Returns the size of the matrix: `(row, col)`.
    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the number of rows in the matrix.
    pub fn num_rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the matrix.
    pub fn num_cols(&self) -> usize {
        self.cols
    }

    /// Returns the specified row of the matrix.
    pub fn row(&self, idx: usize) -> &Row {
        &self[idx]
    }

    /// Returns the specified column of the matrix.
    pub fn col(&self, idx: usize) -> Col {
        let mut data = vec![];
        for row in &self.data {
            data.push(row[idx]);
        }
        Col::new(data)
    }

    /// Multiply each element of the row by a scalar.
    pub fn scale(&self, scalar: f32) -> Self {
        let mut scaled_rows = vec![];
        for val in &self.data {
            scaled_rows.push(val.scale(scalar));
        }
        Self {
            data: scaled_rows,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Multiplies `self` with the given column.
    pub fn mul_col(&self, other: &Col) -> Col {
        // Input validation
        if self.cols != other.data.len() {
            panic!(
                "InvalidShape: `other` must have the same length as `self`'s columns ({})",
                self.data.len()
            )
        }

        let mut row_data = vec![];
        for row in &self.data {
            row_data.push(row * other);
        }
        Col::new(row_data)
    }

    /// Performs matrix multiplication of `self` and `other`.
    pub fn mul_mat(&self, other: &Self) -> Self {
        let mut rows = vec![];
        for row in &self.data {
            rows.push(row * other);
        }

        Self {
            data: rows,
            rows: self.rows,
            cols: other.cols,
        }
    }

    /// Divides the scalar by each element in the column.
    pub fn div_scalar(&mut self, scalar: f32) -> Self {
        let mut scaled = vec![];
        for row in &self.data {
            scaled.push(row.div_scalar(scalar))
        }
        Self {
            data: scaled,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Pushes the given row to the end of the matrix.
    pub fn push_row(&mut self, row: Row) {
        // Input validation
        {
            if self.cols == 0 {
                self.cols = row.data.len();
            } else {
                if row.data.len() != self.cols {
                    panic!("InvalidRow: The row must have {} elements", self.cols)
                }
            }
        }

        self.data.push(row);
        self.rows += 1;
    }

    /// Pushes the given column to the end of the matrix.
    pub fn push_col(&mut self, col: Col) {
        // Input validation
        {
            if self.rows == 0 {
                self.rows = col.data.len();
            } else {
                if col.data.len() != self.rows {
                    panic!("InvalidRow: The row must have {} elements", self.cols)
                }
            }
        }

        for (i, row) in self.data.iter_mut().enumerate() {
            row.data.push(col[i]);
        }
        self.cols += 1;
    }
}

impl Index<usize> for Matrix {
    type Output = Row;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl std::fmt::Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Matrix ({} x {}) [\n", self.rows, self.cols))?;
        for row in &self.data {
            f.write_fmt(format_args!("\t{:?}\n", row))?;
        }
        f.write_str("]")
    }
}

impl Mul<Matrix> for f32 {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Self::Output {
        rhs.scale(self)
    }
}

impl Mul<&Matrix> for f32 {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        rhs.scale(self)
    }
}

impl Mul<f32> for Matrix {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

impl Mul<f32> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

impl Mul<Col> for Matrix {
    type Output = Col;

    fn mul(self, rhs: Col) -> Self::Output {
        self.mul_col(&rhs)
    }
}

impl Mul<&Col> for &Matrix {
    type Output = Col;

    fn mul(self, rhs: &Col) -> Self::Output {
        self.mul_col(rhs)
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Self;

    fn mul(self, rhs: Matrix) -> Self::Output {
        self.mul_mat(&rhs)
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        self.mul_mat(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn create_new() {
        let v = vec![1., 2., 3., 4., 5., 6.];
        let mat = Matrix::from_slice(2, 3, &v);
        dbg!(mat);

        let mut mat = Matrix::new();
        mat.push_row(Row::new(vec![1., 2., 3.]));
        dbg!(mat);

        let mut mat = Matrix::with_capacity(3);
        mat.push_row(Row::new(vec![1., 2., 3.]));
        mat.push_row(Row::new(vec![4., 5., 6.]));
        mat.push_col(Col::new(vec![7., 8.]));
        dbg!(&mat);
        mat[1] = Row::new(vec![0., 0., 0., 0.]);
        dbg!(&mat);
    }

    #[test]
    fn mul_row() {
        let row = Row::new(vec![1., 2., 3.]);

        // Scale
        {
            let scaled = row.clone() * 2_f32;
            assert_eq!(scaled, Row::new(vec![2., 4., 6.]));
        }

        // Dot product
        {
            let dot = &row * &row;
            assert_eq!(dot, 14.);
        }

        // Multiply Column
        {
            let dot = &row * &Col::new(vec![1., 2., 3.]);
            assert_eq!(dot, 14.);
        }

        // Multiply Matrix
        {
            let prod = &row * &Matrix::from_slice(3, 2, &vec![1., 4., 2., 5., 3., 6.]);
            assert_eq!(prod, Row::new(vec![14., 32.]));
        }
    }

    #[test]
    fn mul_col() {
        let col = Col::new(vec![1., 2., 3.]);

        // Scale
        {
            let scaled = col.clone() * 2_f32;
            assert_eq!(scaled, Col::new(vec![2., 4., 6.]));
        }

        // Dot product
        {
            let dot = &col * &col;
            assert_eq!(dot, 14.);
        }

        // Multiply Row
        {
            let prod = &col * &Row::new(vec![1., 2., 3.]);
            assert_eq!(
                prod,
                Matrix::from_slice(3, 3, &vec![1., 2., 3., 2., 4., 6., 3., 6., 9.])
            );
        }
    }

    #[test]
    fn mul_mat() {
        let mat = Matrix::from_slice(3, 2, &vec![1., 2., 3., 4., 5., 6.]);

        // Scale
        {
            let scaled = mat.clone() * 2_f32;
            assert_eq!(
                scaled,
                Matrix::from_slice(3, 2, &vec![2., 4., 6., 8., 10., 12.])
            );
        }

        // Multiply Matrix
        {
            let prod = &mat * &Matrix::from_slice(2, 3, &vec![1., 2., 3., 4., 5., 6.]);
            assert_eq!(
                prod,
                Matrix::from_slice(3, 3, &vec![9., 12., 15., 19., 26., 33., 29., 40., 51.])
            );
        }

        // Multiply Col
        {
            let col = &mat * &Col::new(vec![1., 2.]);
            assert_eq!(col, Col::new(vec![5., 11., 17.]));
        }
    }
}
