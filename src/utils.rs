use crate::field::{F256Point, F256};
use crate::key::{SALT_LEN_BYTES, SK_SEED_BYTES, PK_SEED_BYTES, PublicKey, PrivateKey};
use aes::{Aes128, cipher::{KeyInit, generic_array::GenericArray, BlockEncrypt}};
use ndarray::{Array2, Array3, stack, ArrayView2};
use num_traits::Zero;
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};
use std::{ops::Add, iter::zip};

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

pub fn aes_hash(seed: &[u8; 16], out: usize) -> Vec<F256Point> {
    let key = GenericArray::from(*seed);
    let cipher = Aes128::new(&key);
    let mut result : Vec<F256Point> = Vec::new();
    let mut bytes_remaining = out;
    let mut ctr: u128 = 0;
    while bytes_remaining > 0 {
        let mut block = GenericArray::from(ctr.to_be_bytes());
        /* 
         * Note that in the original algorithms, the authors
         * omit the shift rows step of AES. Here, we don't
         * omit that step since re-implementing AES is a pain.
         */
        cipher.encrypt_block(&mut block);
        ctr += 1;
        bytes_remaining = if bytes_remaining >= 16 { bytes_remaining - 16 } else { 0 };
        for byte in block.iter() {
            result.push(F256.make_point(*byte));
        }
    }
    result
}

pub fn expand_v(
    message: &[u8],
    salt: &[u8; SALT_LEN_BYTES],
    seed: &[u8; SK_SEED_BYTES],
    counter: u8,
    n: usize,
    m: usize
) -> Vec<F256Point> {
    let mut concatenated: Vec<u8> = message.to_owned();
    concatenated.reserve_exact(16 + SK_SEED_BYTES + 1);
    concatenated.extend(salt.iter());
    concatenated.extend(seed.iter());
    concatenated.push(counter);
    hash(&concatenated, n - m)
}

pub fn expand_sk(
    seed: &[u8; SK_SEED_BYTES],
    n: usize,
    m: usize
) -> Array2<F256Point> {
    Array2::from_shape_vec((n - m, m), hash(seed, (n - m) * m)).unwrap()
}

pub fn expand_p(
    seed: &[u8; 16],
    n: usize,
    m: usize
) -> (Array3<F256Point>, Array3<F256Point>) {
    let p1_num_elems = m * (n - m) * (n - m + 1) / 2;  // n - m  x  n - m  but upper triangular
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

pub type ExpandedSecretKey = (
    [u8; SK_SEED_BYTES],
    Array2<F256Point>,
    Array3<F256Point>,
    Vec<Array2<F256Point>>,
);

pub fn key_gen(
    seed_sk: [u8; SK_SEED_BYTES],
    seed_pk: [u8; PK_SEED_BYTES],
    n: usize,
    m: usize,
) -> Result<(PrivateKey, PublicKey), String> {
    let o = expand_sk(&seed_sk, n, m);
    let pmats = expand_p(&seed_pk, n, m);
    let mut p: Array3<F256Point> = Array3::zeros((0, n, n));
    let mut smats: Vec<Array2<F256Point>> = Vec::new();

    for (p1, p2) in zip(pmats.0.outer_iter(), pmats.1.outer_iter()) {
        let ot = o.t();
        let p3 = upper_triangular(
            &(&ot.dot(&p1).dot(&o) - &ot.dot(&p2))
        ).unwrap();
        let ptop = ndarray::concatenate(
            ndarray::Axis(1),
            &[p1, p2]
        );
        let pbot = ndarray::concatenate(
            ndarray::Axis(1), 
            &[Array2::zeros((m, n - m)).view(), p3.view()]
        );
        let pslice = ndarray::concatenate(
            ndarray::Axis(0),
            &[ptop.unwrap().view(), pbot.unwrap().view()]
        ).unwrap().insert_axis(ndarray::Axis(0));
        p = ndarray::concatenate(
            ndarray::Axis(0),
            &[p.view(), pslice.view()]
        ).unwrap();
        smats.push(&(&p1 + &p1.t()).dot(&o) + &p2);
    }

    let smat_views: Vec<ArrayView2<F256Point>> = smats.iter().map(|mat| mat.view()).collect();
    let s: Array3<F256Point> = stack(ndarray::Axis(0), &smat_views).unwrap();

    let esk = match PrivateKey::from_components(
        seed_sk, o, pmats.0, s, n, m
    ) {
        Ok(k) => k,
        Err(err ) => return Err("Error creating private key: \n".to_owned() + &err),
    };

    let epk = match PublicKey::from_p(p) {
        Ok(k) => k,
        Err(err) => return Err("Error creating public key: \n".to_owned() + &err),
    };

    Ok((esk, epk))
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

    #[test]
    fn test_key_gen() {
        use hex::decode;
        let raw_seed = decode("061550234D158C5EC95595FE04EF7A25767F2E24CC2BC479D09D86DC9ABCFDE7056A8C266F9EF97ED08541DBD2E1FFA1").unwrap();
        let mut seed_sk: [u8; SK_SEED_BYTES] = [0u8; SK_SEED_BYTES];
        let mut seed_pk: [u8; PK_SEED_BYTES] = [0u8; PK_SEED_BYTES];
        let mut i: usize = 0;
        for b in &raw_seed[0..SK_SEED_BYTES] {
            seed_sk[i] = *b;
            i += 1;
        }
        i = 0;
        for b in &raw_seed[SK_SEED_BYTES..] {
            seed_pk[i] = *b;
            i += 1;
        }
        let pmats = expand_p(&seed_pk, 112, 44);
        let (esk, epk) = key_gen(seed_sk, seed_pk, 112, 44).unwrap();
    }
}
