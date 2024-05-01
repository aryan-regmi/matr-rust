use core::panic;
use std::ops::{Index, IndexMut, Mul};

/// Represents a row of a matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    data: Vec<f32>,
}

impl Row {
    fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Multiply each element of the row by a scalar.
    pub fn scale(&mut self, scalar: f32) {
        for val in &mut self.data {
            *val = *val * scalar;
        }
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

    /// Multiplies `self` with the given column.
    pub fn mul_col(&self, other: &Col) -> f32 {
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
    pub fn mul_mat(&self, other: &Matrix) -> Self {
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

impl Mul<f32> for Row {
    type Output = Self;

    fn mul(mut self, rhs: f32) -> Self::Output {
        self.scale(rhs);
        self
    }
}

impl Mul<Row> for f32 {
    type Output = Row;

    fn mul(self, mut rhs: Row) -> Self::Output {
        rhs.scale(self);
        rhs
    }
}

impl Mul<f32> for &mut Row {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs);
        self
    }
}

impl<'a> Mul<&'a mut Row> for f32 {
    type Output = &'a Row;

    fn mul(self, rhs: &'a mut Row) -> Self::Output {
        rhs.scale(self);
        rhs
    }
}

impl Mul<Row> for Row {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        self.dot(&rhs)
    }
}

impl<'a> Mul<&'a Row> for &'a Row {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        self.dot(&rhs)
    }
}

impl Mul<Col> for Row {
    type Output = f32;

    fn mul(self, rhs: Col) -> Self::Output {
        self.mul_col(&rhs)
    }
}

impl<'a> Mul<&'a Col> for &'a Row {
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

impl<'a> Mul<&'a Matrix> for &'a Row {
    type Output = Row;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        self.mul_mat(&rhs)
    }
}

/// Represents a column of a matrix.
#[derive(Debug)]
pub struct Col {
    data: Vec<f32>,
}

impl Col {
    fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Multiply each element of the row by a scalar.
    pub fn scale(&mut self, scalar: f32) {
        for val in &mut self.data {
            *val = *val * scalar;
        }
    }

    // /// Calculates the dot prodcut of `self` and `other`.
    pub fn dot(&self, other: &Self) -> f32 {
        //     // Input validation
        //     if self.data.len() != other.data.len() {
        //         panic!(
        //             "InvalidShape: `other` must have the same length as `self` ({})",
        //             self.data.len()
        //         )
        //     }
        //
        //     let mut sum = 0.0;
        //     for i in 0..self.data.len() {
        //         sum += self[i] * other[i];
        //     }
        //     sum
        todo!()
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

        // Matrix::borrow

        todo!()

        // let mut sum = 0.0;
        // for i in 0..self.data.len() {
        //     sum += self[i] * other[i];
        // }
        // sum
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

impl Mul<f32> for Col {
    type Output = Self;

    fn mul(mut self, rhs: f32) -> Self::Output {
        self.scale(rhs);
        self
    }
}

impl Mul<Col> for f32 {
    type Output = Col;

    fn mul(self, mut rhs: Col) -> Self::Output {
        rhs.scale(self);
        rhs
    }
}

impl Mul<f32> for &mut Col {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs);
        self
    }
}

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

    pub fn from_rows(row_vec: Vec<Row>) -> Self {
        let m = row_vec.len();
        let n = row_vec[0].data.len();

        // Input validation
        {
            for (i, row) in row_vec.iter().enumerate() {
                if row.data.len() != n {
                    panic!("InvalidRows: All rows must have the same number of elements ({}), but row {} in the given vector has {} elements", n, i,row.data.len())
                }
            }
        }

        todo!()
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
        f.write_fmt(format_args!("Matrix ({} x {}) \n[", self.rows, self.cols))?;
        for i in 0..self.rows {
            f.write_str("\t\n")?;
            for j in 0..self.cols {
                f.write_fmt(format_args!(" {} ", self[i][j]))?;
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
    fn create_from_slice() {
        let v = vec![1., 2., 3., 4., 5., 6.];
        let mat = Matrix::from_slice(2, 3, &v);
        dbg!(&mat);
        dbg!(&mat.row(0));
        dbg!(&mat.col(0));
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
}
