use ndarray::{concatenate, s, Array1, ArrayView1, ArrayView2, Axis};
use num_traits::{abs, One, Signed, Zero};
use std::{
    cmp::min,
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

pub fn determinant<
    T: Zero + Clone + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
>(
    a: &ArrayView2<T>,
) -> Result<T, String> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(format!(
            "{} x {} matrix must be square.",
            shape[0], shape[1]
        ));
    }

    if shape[0] < 2 {
        return Err(format!(
            "{} x {} matrix is too small. It must be at least 2 x 2",
            shape[0], shape[1]
        ));
    }

    if shape[0] == 2 {
        Ok(a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]])
    } else {
        let mut result = T::zero();
        for i in 0..shape[0] {
            let row_cut = a.slice(s![1.., ..]);
            let sliced = concatenate(
                Axis(1),
                &[row_cut.slice(s![.., ..i]), row_cut.slice(s![.., (i + 1)..])],
            )
            .unwrap();
            let subdet = a[[0, i]] * determinant(&sliced.view()).unwrap();
            if i % 2 == 0 {
                result = result + subdet;
            } else {
                result = result - subdet;
            }
        }
        Ok(result)
    }
}

pub fn solve<
    T: 'static
        + Zero
        + One
        + Clone
        + Copy
        + Debug
        + PartialOrd
        + Signed
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>,
>(
    a: &ArrayView2<T>,
    y: &ArrayView1<T>,
) -> Option<Array1<T>> {
    let shape = a.shape();
    let m = shape[0];
    let n = shape[1];
    let mut target = y.to_owned();
    let mut rows: Vec<Array1<T>> = a.rows().into_iter().map(|r| r.to_owned()).collect();

    let mut pivot_row = 0;
    let mut pivot_col = 0;

    while pivot_row < m && pivot_col < n {
        let mut i_max = 0;
        let mut abs_max: T = T::zero();
        for i in pivot_row..m {
            let v = abs(rows.get(i).unwrap()[[pivot_col]]);
            if v >= abs_max {
                abs_max = v;
                i_max = i;
            }
        }
        if T::is_zero(&rows.get(i_max).unwrap()[[pivot_col]]) {
            pivot_col += 1;
        } else {
            rows.swap(pivot_row, i_max);
            target.swap(pivot_row, i_max);

            for i in (0..pivot_row).chain((pivot_row + 1)..m) {
                let f =
                    rows.get(i).unwrap()[[pivot_col]] / rows.get(pivot_row).unwrap()[[pivot_col]];

                rows.get_mut(i).unwrap()[[pivot_col]] = T::zero();
                for j in (pivot_col + 1)..n {
                    rows.get_mut(i).unwrap()[[j]] =
                        rows.get(i).unwrap()[[j]] - rows.get(pivot_row).unwrap()[[j]] * f;
                }

                target[[i]] = target[[i]] - target[[pivot_row]] * f;
            }

            pivot_row += 1;
            pivot_col += 1;
        }
    }

    let mut rank = 0;
    for row_i in 0..m {
        let mut scol: i64 = -1;
        {
            let row = rows.get_mut(row_i).unwrap();
            for col in 0..n {
                if !T::is_zero(&row[[col]]) {
                    scol = col as i64;
                    target[[row_i]] = target[[row_i]] / row[[col]];
                    row[[col]] = T::one();
                    break;
                }
            }
        }
        if scol >= 0 {
            rows.swap(min(scol as usize, m - 1), row_i);
            target.swap(min(scol as usize, m - 1), row_i);
            rank += 1;
        }
    }

    if n == m && rank == n {
        Some(target)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_determinant_simple_success() {
        let a: Array2<i32> = Array2::from_shape_vec((2, 2), (1..5).collect()).unwrap();
        assert_eq!(determinant(&a.view()).unwrap(), -2);
    }

    #[test]
    fn test_determinant_success() {
        let mut a: Array2<i32> = Array2::from_shape_vec((3, 3), (1..10).collect()).unwrap();
        assert_eq!(determinant(&a.view()).unwrap(), 0);

        a = Array2::from_shape_vec((4, 4), (1..17).collect()).unwrap();
        assert_eq!(determinant(&a.view()).unwrap(), 0);
    }

    #[test]
    fn test_solve_simple_success() {
        let a: Array2<f32> =
            Array2::from_shape_vec((2, 2), (1..5).map(|v| v as f32).collect()).unwrap();
        let y: Array1<f32> = Array1::from_iter((1..3).map(|v| v as f32));
        let x = solve(&a.view(), &y.view()).unwrap();
        assert!(a.dot(&x).abs_diff_eq(&y, 1e-6));
    }

    #[test]
    fn test_solve_success() {
        let a: Array2<f32> = Array2::from_shape_vec(
            (3, 3),
            vec![1f32, 2f32, 3f32, 2f32, -1f32, 1f32, 3f32, 0f32, -1f32],
        )
        .unwrap();
        let y: Array1<f32> = Array1::from_iter(vec![9f32, 8f32, 3f32]);
        let x = solve(&a.view(), &y.view()).unwrap();
        assert!(a.dot(&x).abs_diff_eq(&y, 1e-6));
    }
}
