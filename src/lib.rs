use core::panic;
use std::{
    mem::ManuallyDrop,
    ops::{Index, IndexMut},
    ptr::NonNull,
};

// TODO: Add arithmetic operators!
//
// TODO: Add func to solve Ax = b
//  - Add matrix decomp/determinant funcs

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
    pub fn mul(&self, other: &Self) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_create_matrix() {
        let mat = Matrix::new(2, 3);
        assert_eq!(mat.num_rows(), 2);
        assert_eq!(mat.num_cols(), 3);
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

        let mult = mat1.mul(&mat2);

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
}
