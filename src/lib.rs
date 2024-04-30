use std::ops::{Index, IndexMut};

/// Represents a row of a matrix.
#[derive(Debug)]
pub struct Row {
    data: Vec<f32>,
}

impl Row {
    fn new(data: Vec<f32>) -> Self {
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

/// Represents a column of a matrix.
#[derive(Debug)]
pub struct Col {
    data: Vec<f32>,
}

impl Col {
    fn new(data: Vec<f32>) -> Self {
        Self { data }
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
}
