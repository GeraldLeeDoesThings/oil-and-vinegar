use std::iter::zip;

use ndarray::{Array3, Array2, s, ArrayView2};
use crate::{field::F256Point, field::F256, utils::upper_triangular};


pub const PK_SEED_LEN: usize = 128;
pub const SK_SEED_LEN: usize = 256;
pub const SALT_LEN: usize = 128;

pub const PK_SEED_BYTES: usize = PK_SEED_LEN / 8;
pub const SK_SEED_BYTES: usize = SK_SEED_LEN / 8;
pub const SALT_LEN_BYTES: usize = SALT_LEN / 8;


pub struct PublicKey {

    p_matrix: Array3<F256Point>,
    n: usize,
    m: usize,

}


pub struct PrivateKey {

    seed: [u8; SK_SEED_BYTES],
    o_matrix: Array2<F256Point>,
    p_matrix_1: Array3<F256Point>,
    s_matrix: Array3<F256Point>,
    n: usize,
    m: usize,

}


impl PublicKey {

    pub fn from_components(p1: &Array3<F256Point>, p2: &Array3<F256Point>, o: &Array2<F256Point>, n: usize, m: usize) -> Result<Self, String> {
        if p1.shape() != &[m, n - m, n - m] {
            return Err(format!("P1 is misshapen! With m={}, n={}, expected shape {:?}, but found shape {:?}", m, n, [m, n - m, n - m], p1.shape()));
        }
        if p2.shape() != &[m, n - m, m] {
            return Err(format!("P2 is misshapen! With m={}, n={}, expected shape {:?}, but found shape {:?}", m, n, [m, n - m, m], p2.shape()));
        }
        if o.shape() != &[n - m, m] {
            return Err(format!("O is misshapen! With m={}, n={}, expected shape {:?}, but found shape {:?}", m, n, [n - m, m], o.shape()));
        }

        let mut p: Array3<F256Point> = Array3::zeros((0, n, n));

        let ot = o.t();
        for (p1, p2) in zip(p1.outer_iter(), p2.outer_iter()) {
            
            let p3 = upper_triangular(
                &(&ot.dot(&p1).dot(o) - &ot.dot(&p2))
            );
            let ptop = ndarray::concatenate(
                ndarray::Axis(1),
                &[p1, p2]
            );
            let pbot = ndarray::concatenate(
                ndarray::Axis(1), 
                &[Array2::zeros((m, n - m)).view(), p3.unwrap().view()]
            );
            let pslice = ndarray::concatenate(
                ndarray::Axis(0),
                &[ptop.unwrap().view(), pbot.unwrap().view()]
            ).unwrap().insert_axis(ndarray::Axis(0));
            p = ndarray::concatenate(
                ndarray::Axis(0),
                &[p.view(), pslice.view()]
            ).unwrap();
        }

        return Ok(PublicKey { p_matrix: p, n, m });
    }

    pub fn from_elems(elems: Vec<F256Point>, n: usize, m: usize) -> Result<Self, String> {
        let mut p: Array3<F256Point> = Array3::zeros((m, n, n));
        let mut next_elem_index: usize = 0;
        for row in 0..(n - m) {
            for col in row..(n - m) {
                for mat in 0..m {
                    p[[mat, row, col]] = *elems.get(next_elem_index).ok_or("Ran out of elements!")?;
                    next_elem_index += 1;
                }
            }
        }
    
        for row in 0..(n - m) {
            for col in (n - m)..n {
                for mat in 0..m {
                    p[[mat, row, col]] = *elems.get(next_elem_index).ok_or("Ran out of elements!")?;
                    next_elem_index += 1;
                }
            }
        }

        for row in (n - m)..n {
            for col in row..n {
                for mat in 0..m {
                    p[[mat, row, col]] = *elems.get(next_elem_index).ok_or("Ran out of elements!")?;
                    next_elem_index += 1;
                }
            }
        }

        return Ok(PublicKey { p_matrix: p, n, m });
    }

    pub fn from_bytes(bytes: Vec<u8>, n: usize, m: usize) -> Result<Self, String> {
        return PublicKey::from_elems(bytes.iter().map(|b| F256.make_point(*b)).collect(), n, m);
    }

    pub fn from_p(p_matrix: Array3<F256Point>) -> Result<Self, String> {
        let shape = p_matrix.shape();
        if shape.len() != 3 {
            return Err(format!("Matrix P must have exactly 3 dimensions! Found {} dimensions instead.", shape.len()))
        }
        if shape[1] != shape[2] {
            return Err(format!("Last two dimensions of matrix P must be the same! Instead found {} and {}.", shape[1], shape[2]));
        }

        let m = shape[0];
        let n = shape[1];

        if m >= n {
            return Err(format!("First dimension of matrix P must be smaller than the other two! Found {} which is >= to {}", m, n));
        }

        Ok(PublicKey { p_matrix, n: n, m: m })
    }

    pub fn p1(&self, i: usize) -> ArrayView2<F256Point> {
        self.p_matrix.index_axis(ndarray::Axis(0), i).slice_move(s![0..(self.n - self.m), 0..(self.n - self.m)])
    }

    pub fn p2(&self, i: usize) -> ArrayView2<F256Point> {
        self.p_matrix.index_axis(ndarray::Axis(0), i).slice_move(s![0..(self.n - self.m), (self.n - self.m)..self.n])
    }

    pub fn p3(&self, i: usize) -> ArrayView2<F256Point> {
        self.p_matrix.index_axis(ndarray::Axis(0), i).slice_move(s![(self.n - self.m)..self.n, (self.n - self.m)..self.n])
    }

    pub fn p(&self, i: usize) -> ArrayView2<F256Point> {
        self.p_matrix.index_axis(ndarray::Axis(0), i)
    }

}


impl PrivateKey {

    pub fn from_components(
        seed: [u8; SK_SEED_BYTES],
        o_matrix: Array2<F256Point>,
        p_matrix_1: Array3<F256Point>,
        s_matrix: Array3<F256Point>,
        n: usize,
        m: usize,
    ) -> Result<Self, String> {

        if n <= m {
            return Err(format!("Must have n > m, but found n: {}, m: {}", n, m));
        }

        if o_matrix.shape() != &[n - m, m] {
            return Err(format!("Shape of O matrix is wrong! Expected {:?}, but found {:?}", &[n - m, m], o_matrix.shape()));
        }

        if p_matrix_1.shape() != &[m, n - m, n - m] {
            return Err(format!("Shape of P1 matrix is wrong! Expected {:?}, but found {:?}", &[m, n - m, n - m], p_matrix_1.shape()));
        }

        if s_matrix.shape() !=  &[m, n - m, m] {
            return Err(format!("Shape of S matrix is wrong! Expected {:?}, but found {:?}", &[m, n - m, m], s_matrix.shape()));
        }

        Ok(PrivateKey { seed, o_matrix, p_matrix_1, s_matrix, n, m })
    }

}


#[cfg(test)]
mod tests {
    use crate::utils::{expand_p, expand_sk};
    use super::*;

    #[test]
    fn test_public_key_from_components() {
        let seed_sk: [u8; SK_SEED_BYTES] = [0u8; SK_SEED_BYTES];
        let seed_pk: [u8; PK_SEED_BYTES] = [0u8; PK_SEED_BYTES];

        let o = expand_sk(&seed_sk, 112, 44);
        let ot = o.t();
        let pmats = expand_p(&seed_pk, 112, 44);
        let pk = PublicKey::from_components(&pmats.0, &pmats.1, &o, 112, 44).unwrap();

        for _round in 0..2 {
            for i in 0..44 {
                let p1 = pk.p1(i);
                let p2 = pk.p2(i);
                
                assert_eq!(pmats.0.index_axis(ndarray::Axis(0), i), p1);
                assert_eq!(pmats.1.index_axis(ndarray::Axis(0), i), p2);

                let p3 = upper_triangular(
                    &(&ot.dot(&p1).dot(&o) - &ot.dot(&p2))
                ).unwrap();

                assert_eq!(p3, pk.p3(i));
            }
        }
    }

    #[test]
    fn test_public_key_from_elems() {
        let seed_sk: [u8; SK_SEED_BYTES] = [0u8; SK_SEED_BYTES];
        let seed_pk: [u8; PK_SEED_BYTES] = [0u8; PK_SEED_BYTES];

        let o: Array2<F256Point> = expand_sk(&seed_sk, 112, 44);
        let ot = o.t();
        let pmats = expand_p(&seed_pk, 112, 44);
        // let pk = PublicKey::<112, 44>::from_components(&pmats.0, &pmats.1, &o).unwrap();

        let mut p1_vec: Vec<ArrayView2<F256Point>> = Vec::new();
        let mut p2_vec: Vec<ArrayView2<F256Point>> = Vec::new();
        let mut p3_vec: Vec<Array2<F256Point>> = Vec::new();
        let mut elem_vec: Vec<F256Point> = Vec::new();

        for i in 0..44 {
            let p1 = pmats.0.index_axis(ndarray::Axis(0), i);
            let p2 = pmats.1.index_axis(ndarray::Axis(0), i);
            let p3 = upper_triangular(
                &(&ot.dot(&p1).dot(&o) - &ot.dot(&p2))
            ).unwrap();

            p1_vec.push(p1);
            p2_vec.push(p2);
            p3_vec.push(p3);
        }

        for row in 0..(112 - 44) {
            for col in row..(112 - 44) {
                for i in 0..44 {
                    elem_vec.push(p1_vec.get(i).unwrap()[[row, col]]);
                }
            }
        }

        for row in 0..(112 - 44) {
            for col in 0..44 {
                for i in 0..44 {
                    elem_vec.push(p2_vec.get(i).unwrap()[[row, col]]);
                }
            }
        }

        for row in 0..44 {
            for col in row..44 {
                for i in 0..44 {
                    elem_vec.push(p3_vec.get(i).unwrap()[[row, col]]);
                }
            }
        }

        let pk = PublicKey::from_elems(elem_vec, 112, 44).unwrap();

        for _round in 0..2 {
            for i in 0..44 {
                let p1 = p1_vec.get(i).unwrap();
                let p2 = p2_vec.get(i).unwrap();
                let p3 = p3_vec.get(i).unwrap();

                assert_eq!(p1, pk.p1(i));
                assert_eq!(p2, pk.p2(i));
                assert_eq!(p3, pk.p3(i));
            }
        }
    }

}
