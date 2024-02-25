use crate::field::F256Point;
use crate::hazmat::hash::{aes_hash, hash};
use crate::key::{SALT_LEN_BYTES, SK_SEED_BYTES};
use ndarray::{Array1, Array2, Array3};

pub fn expand_v(
    message: &[u8],
    salt: &[u8; SALT_LEN_BYTES],
    seed: &[u8; SK_SEED_BYTES],
    counter: u8,
    n: usize,
    m: usize,
) -> Array1<F256Point> {
    let mut concatenated: Vec<u8> = message.to_owned();
    concatenated.reserve_exact(16 + SK_SEED_BYTES + 1);
    concatenated.extend(salt.iter());
    concatenated.extend(seed.iter());
    concatenated.push(counter);
    Array1::from_vec(hash(&concatenated, n - m))
}

pub fn expand_sk(seed: &[u8; SK_SEED_BYTES], n: usize, m: usize) -> Array2<F256Point> {
    Array2::from_shape_vec((n - m, m), hash(seed, (n - m) * m)).unwrap()
}

pub fn expand_p(seed: &[u8; 16], n: usize, m: usize) -> (Array3<F256Point>, Array3<F256Point>) {
    let p1_num_elems = m * (n - m) * (n - m + 1) / 2; // n - m  x  n - m  but upper triangular
    let p2_num_elems = usize::pow(m, 2) * (n - m); // n - m  x  m
    let p_elems = aes_hash(seed, p1_num_elems + p2_num_elems);
    let mut p1_matricies = Array3::zeros((m, n - m, n - m));
    let mut p2_matricies = Array3::zeros((m, n - m, m));
    let mut next_elem_index = 0;

    for row in 0..(n - m) {
        for col in row..(n - m) {
            for mat in 0..m {
                p1_matricies[[mat, row, col]] = p_elems[next_elem_index];
                next_elem_index += 1;
            }
        }
    }

    for row in 0..(n - m) {
        for col in 0..m {
            for mat in 0..m {
                p2_matricies[[mat, row, col]] = p_elems[next_elem_index];
                next_elem_index += 1;
            }
        }
    }
    (p1_matricies, p2_matricies)
}
