use crate::field::{F256Point, F256};
use ndarray::{Array2, Array3};
use num_traits::Zero;
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::ops::Add;

pub fn hash(message: &[u8], m: usize) -> Vec<F256Point> {
    let mut hasher = Shake256::default();
    hasher.update(message);
    let mut reader = hasher.finalize_xof();
    let mut result: Vec<F256Point> = Vec::with_capacity(m);
    let mut hashed_byte = [0u8; 1];
    for _ in 0..m {
        reader.read(&mut hashed_byte);
        result.push(F256.make_point(hashed_byte[0]));
    }
    result
}

pub fn expand_v<const SEED_SK_LEN: usize>(
    message: &[u8],
    salt: &[u8; 16],
    seed: &[u8; SEED_SK_LEN],
    counter: u8,
    n: usize,
    m: usize
) -> Vec<F256Point> {
    let mut concatenated: Vec<u8> = message.to_owned();
    concatenated.reserve_exact(16 + SEED_SK_LEN + 1);
    concatenated.extend(salt.iter());
    concatenated.extend(seed.iter());
    concatenated.push(counter);
    hash(&concatenated, n - m)
}

pub fn expand_sk<const SEED_SK_LEN: usize>(
    seed: &[u8; SEED_SK_LEN],
    n: usize,
    m: usize
) -> Array2<F256Point> {
    Array2::from_shape_vec((m, n - m), hash(seed, m * (n - m))).unwrap()
}

pub fn expand_p<const SEED_PK_LEN: usize>(
    seed: &[u8; SEED_PK_LEN],
    n: usize,
    m: usize
) -> (Array3<F256Point>, Array3<F256Point>) {
    let p1_num_elems = m * (n - m) * (n - m + 1) / 2;  // n - m  x  n - m  but upper triangular
    let p2_num_elems = usize::pow(m, 2) * (n - m); // n - m  x  m
    let p1_elems = hash(seed, p1_num_elems);
    let mut p1_matricies = Array3::zeros((m, n - m, n - m));
    let mut p1_next_elem_index = 0;
    
    for row in 0..(n - m) {
        for col in row..(n - m) {
            for mat in 0..m {
                p1_matricies[[mat, row, col]] = p1_elems[p1_next_elem_index];
                p1_next_elem_index += 1;
            }
        }
    }
    let p2_matricies = Array3::from_shape_vec((m, n - m, m), hash(seed, p2_num_elems)).unwrap();
    (p1_matricies, p2_matricies)
}

pub fn upper_triangular<T: Add + Clone + Copy + Zero>(matrix: &Array2<T>) -> Result<Array2<T>, String> {
    let dim = matrix.dim();
    if !matrix.is_square() {
        return Err(format!("Non-square matrix passed to 'upper' with dimensions {} x {}.", dim.0, dim.1));
    }

    // A^T = -A
    // (M' - M)^T = M - M'
    // M' = M - (M' - M)^T
    // M' = M - (M'^T - M^T)
    // M' = M - M'^T + M^T
    // M' + M'^T = M + M^T
    let mut result: Array2<T> = Array2::zeros(dim);

    for row in 0..(dim.0) {
        for col in row..(dim.0) {
            if row == col {
                result[[row, col]] = matrix[[row, col]];
            }
            else {
                result[[row, col]] = matrix[[row, col]] + matrix[[col, row]];
            }
        }
    }

    return Ok(result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upper_triangular_success() {
        let sample = Array2::from_shape_vec(
            (3, 3),
            (1..10).collect()
        ).unwrap();
        let sample_prime = upper_triangular(&sample).unwrap();
        let subtracted = sample - sample_prime;
        let should_be_zero = subtracted.clone().reversed_axes() + subtracted;
        
        let is_all_zero = should_be_zero.mapv(
            |v| v == 0
        ).fold(
            true,
            |n, s| n && *s 
        );

        assert!(is_all_zero);
    }
}
