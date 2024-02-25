use crate::{
    hazmat::expand::{expand_p, expand_sk, expand_v},
    field::{F256Point, F256},
    hazmat::hash::hash_concat,
    linalg::{solve, upper_triangular},
};
use ndarray::{concatenate, s, stack, Array1, Array2, Array3, ArrayView1, ArrayView2};
use rand::{rngs::OsRng, RngCore};
use std::error::Error;
use std::io::{Read, Write};
use std::iter::zip;

pub const PK_SEED_LEN: usize = 128;
pub const SK_SEED_LEN: usize = 256;
pub const SALT_LEN: usize = 128;

pub const PK_SEED_BYTES: usize = PK_SEED_LEN / 8;
pub const SK_SEED_BYTES: usize = SK_SEED_LEN / 8;
pub const SALT_LEN_BYTES: usize = SALT_LEN / 8;

/* uov-Ip */
pub const N_I: usize = 112;
pub const M_I: usize = 44;

/* uov-III */
pub const N_III: usize = 184;
pub const M_III: usize = 72;

/* uov-V */
pub const N_V: usize = 244;
pub const M_V: usize = 96;

#[derive(PartialEq)]
pub struct PublicKey {
    n: usize,
    m: usize,
    p_matrix: Array3<F256Point>,
}

#[derive(PartialEq)]
pub struct PrivateKey {
    n: usize,
    m: usize,
    seed: [u8; SK_SEED_BYTES],
    o_matrix: Array2<F256Point>,
    p_matrix_1: Array3<F256Point>,
    s_matrix: Array3<F256Point>,
}

impl PublicKey {
    pub fn from_components(
        p1: &Array3<F256Point>,
        p2: &Array3<F256Point>,
        o: &Array2<F256Point>,
        n: usize,
        m: usize,
    ) -> Result<Self, String> {
        if p1.shape() != &[m, n - m, n - m] {
            return Err(format!(
                "P1 is misshapen! With m={}, n={}, expected shape {:?}, but found shape {:?}",
                m,
                n,
                [m, n - m, n - m],
                p1.shape()
            ));
        }
        if p2.shape() != &[m, n - m, m] {
            return Err(format!(
                "P2 is misshapen! With m={}, n={}, expected shape {:?}, but found shape {:?}",
                m,
                n,
                [m, n - m, m],
                p2.shape()
            ));
        }
        if o.shape() != &[n - m, m] {
            return Err(format!(
                "O is misshapen! With m={}, n={}, expected shape {:?}, but found shape {:?}",
                m,
                n,
                [n - m, m],
                o.shape()
            ));
        }

        let mut p: Array3<F256Point> = Array3::zeros((0, n, n));

        let ot = o.t();
        for (p1, p2) in zip(p1.outer_iter(), p2.outer_iter()) {
            let p3 = upper_triangular(&(&ot.dot(&p1).dot(o) - &ot.dot(&p2)));
            let ptop = ndarray::concatenate(ndarray::Axis(1), &[p1, p2]);
            let pbot = ndarray::concatenate(
                ndarray::Axis(1),
                &[Array2::zeros((m, n - m)).view(), p3.unwrap().view()],
            );
            let pslice = ndarray::concatenate(
                ndarray::Axis(0),
                &[ptop.unwrap().view(), pbot.unwrap().view()],
            )
            .unwrap()
            .insert_axis(ndarray::Axis(0));
            p = ndarray::concatenate(ndarray::Axis(0), &[p.view(), pslice.view()]).unwrap();
        }

        // TODO: Additional validation

        return Ok(PublicKey { n, m, p_matrix: p });
    }

    pub fn from_elems(elems: Vec<F256Point>, n: usize, m: usize) -> Result<Self, String> {
        let mut p: Array3<F256Point> = Array3::zeros((m, n, n));
        let mut next_elem_index: usize = 0;
        for row in 0..(n - m) {
            for col in row..(n - m) {
                for mat in 0..m {
                    p[[mat, row, col]] =
                        *elems.get(next_elem_index).ok_or("Ran out of elements!")?;
                    next_elem_index += 1;
                }
            }
        }

        for row in 0..(n - m) {
            for col in (n - m)..n {
                for mat in 0..m {
                    p[[mat, row, col]] =
                        *elems.get(next_elem_index).ok_or("Ran out of elements!")?;
                    next_elem_index += 1;
                }
            }
        }

        for row in (n - m)..n {
            for col in row..n {
                for mat in 0..m {
                    p[[mat, row, col]] =
                        *elems.get(next_elem_index).ok_or("Ran out of elements!")?;
                    next_elem_index += 1;
                }
            }
        }

        // TODO: Additional validation

        return Ok(PublicKey { n, m, p_matrix: p });
    }

    pub fn from_bytes(bytes: Vec<u8>, n: usize, m: usize) -> Result<Self, String> {
        return PublicKey::from_elems(bytes.iter().map(|b| F256.make_point(*b)).collect(), n, m);
    }

    pub fn from_p(p_matrix: Array3<F256Point>) -> Result<Self, String> {
        let shape = p_matrix.shape();
        if shape.len() != 3 {
            return Err(format!(
                "Matrix P must have exactly 3 dimensions! Found {} dimensions instead.",
                shape.len()
            ));
        }
        if shape[1] != shape[2] {
            return Err(format!(
                "Last two dimensions of matrix P must be the same! Instead found {} and {}.",
                shape[1], shape[2]
            ));
        }

        let m = shape[0];
        let n = shape[1];

        if m >= n {
            return Err(format!("First dimension of matrix P must be smaller than the other two! Found {} which is >= to {}", m, n));
        }

        // TODO: Additional validation

        Ok(PublicKey {
            n: n,
            m: m,
            p_matrix,
        })
    }

    pub fn p1(&self, i: usize) -> ArrayView2<F256Point> {
        self.p_matrix
            .index_axis(ndarray::Axis(0), i)
            .slice_move(s![0..(self.n - self.m), 0..(self.n - self.m)])
    }

    pub fn p2(&self, i: usize) -> ArrayView2<F256Point> {
        self.p_matrix
            .index_axis(ndarray::Axis(0), i)
            .slice_move(s![0..(self.n - self.m), (self.n - self.m)..self.n])
    }

    pub fn p3(&self, i: usize) -> ArrayView2<F256Point> {
        self.p_matrix
            .index_axis(ndarray::Axis(0), i)
            .slice_move(s![(self.n - self.m)..self.n, (self.n - self.m)..self.n])
    }

    pub fn p(&self, i: usize) -> ArrayView2<F256Point> {
        self.p_matrix.index_axis(ndarray::Axis(0), i)
    }

    pub fn verify(
        &self,
        message: &[u8],
        signature: &ArrayView1<F256Point>,
        salt: &[u8; SALT_LEN_BYTES],
    ) -> bool {
        let t: Array1<F256Point> = Array1::from_vec(hash_concat(&[message, salt], self.m));
        let sigvec: Array1<F256Point> =
            Array1::from_iter((0..self.m).map(|i| signature.t().dot(&self.p(i).dot(signature))));
        t == sigvec
    }

    pub fn save(&self, file: &mut impl Write) -> Result<(), std::io::Error> {
        file.write_all(&(self.n as u64).to_le_bytes())?;
        file.write_all(&(self.m as u64).to_le_bytes())?;
        let p_bytes: Vec<u8> = self.p_matrix.map(|v| v.to_byte()).into_iter().collect();
        file.write_all(&p_bytes)?;
        Ok(())
    }

    pub fn load(file: &mut impl Read) -> Result<PublicKey, Box<dyn Error>> {
        let mut meta_buf = [0u8; 8];
        file.read_exact(&mut meta_buf)?;
        let n: usize = u64::from_le_bytes(meta_buf) as usize;

        file.read_exact(&mut meta_buf)?;
        let m: usize = u64::from_le_bytes(meta_buf) as usize;

        let mut p_buf: Vec<u8> = Vec::new();
        file.read_to_end(&mut p_buf)?;

        let p_elems: Vec<F256Point> = p_buf.iter().map(|b| F256.make_point(*b)).collect();

        let p: Array3<F256Point> = Array3::from_shape_vec((m, n, n), p_elems)?;

        Ok(PublicKey { n, m, p_matrix: p })
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
            return Err(format!(
                "Shape of O matrix is wrong! Expected {:?}, but found {:?}",
                &[n - m, m],
                o_matrix.shape()
            ));
        }

        if p_matrix_1.shape() != &[m, n - m, n - m] {
            return Err(format!(
                "Shape of P1 matrix is wrong! Expected {:?}, but found {:?}",
                &[m, n - m, n - m],
                p_matrix_1.shape()
            ));
        }

        if s_matrix.shape() != &[m, n - m, m] {
            return Err(format!(
                "Shape of S matrix is wrong! Expected {:?}, but found {:?}",
                &[m, n - m, m],
                s_matrix.shape()
            ));
        }

        // TODO: Additional validation

        Ok(PrivateKey {
            n,
            m,
            seed,
            o_matrix,
            p_matrix_1,
            s_matrix,
        })
    }

    pub fn s(&self, i: usize) -> ArrayView2<F256Point> {
        self.s_matrix.index_axis(ndarray::Axis(0), i)
    }

    pub fn sign(&self, message: &[u8]) -> Option<(Array1<F256Point>, [u8; SALT_LEN_BYTES])> {
        let mut salt = [0u8; SALT_LEN_BYTES];
        OsRng.fill_bytes(&mut salt);

        let t: Array1<F256Point> = Array1::from_vec(hash_concat(&[message, &salt], self.m));

        for ctr in 0..=255 {
            let v: Array1<F256Point> = expand_v(message, &salt, &self.seed, ctr, self.n, self.m);
            let mut l_rows: Vec<Array1<F256Point>> = Vec::new();

            for i in 0..self.m {
                l_rows.push(v.t().dot(&self.s(i)));
            }
            let l_views: Vec<ArrayView1<F256Point>> = l_rows.iter().map(|lr| lr.view()).collect();
            let l_matrix: Array2<F256Point> = stack(ndarray::Axis(0), &l_views).unwrap();

            let inv_check = solve(&l_matrix.view(), &Array1::zeros(self.m).view());
            if inv_check.is_some() {
                let y: Array1<F256Point> = Array1::from_iter((0..self.m).map(|i| {
                    v.t()
                        .dot(&self.p_matrix_1.index_axis(ndarray::Axis(0), i))
                        .dot(&v)
                }));
                let x = solve(&l_matrix.view(), &(&t - &y).view()).unwrap();
                let s = concatenate(ndarray::Axis(0), &[v.view(), Array1::zeros(self.m).view()])
                    .unwrap()
                    + concatenate(
                        ndarray::Axis(0),
                        &[self.o_matrix.view(), Array2::eye(self.m).view()],
                    )
                    .unwrap()
                    .dot(&x);
                return Some((s, salt));
            }
        }
        return None;
    }

    pub fn save(&self, file: &mut impl Write) -> Result<(), std::io::Error> {
        file.write_all(&(self.n as u64).to_le_bytes())?;
        file.write_all(&(self.m as u64).to_le_bytes())?;
        file.write_all(&self.seed)?;
        let o_bytes: Vec<u8> = self.o_matrix.map(|v| v.to_byte()).into_iter().collect();
        file.write_all(&o_bytes)?;
        let p_bytes: Vec<u8> = self.p_matrix_1.map(|v| v.to_byte()).into_iter().collect();
        file.write_all(&p_bytes)?;
        let s_bytes: Vec<u8> = self.s_matrix.map(|v| v.to_byte()).into_iter().collect();
        file.write_all(&s_bytes)?;
        Ok(())
    }

    pub fn load(file: &mut impl Read) -> Result<PrivateKey, Box<dyn Error>> {
        let mut meta_buf = [0u8; 8];
        file.read_exact(&mut meta_buf)?;
        let n: usize = u64::from_le_bytes(meta_buf) as usize;

        file.read_exact(&mut meta_buf)?;
        let m: usize = u64::from_le_bytes(meta_buf) as usize;

        let mut seed = [0u8; SK_SEED_BYTES];
        file.read_exact(&mut seed)?;

        let mut mat_buf: Vec<u8> = Vec::new();
        file.read_to_end(&mut mat_buf)?;

        let mut last_index: usize = (n - m) * m;
        let o_buf = &mat_buf[0..((n - m) * m)];

        let p_buf = &mat_buf[last_index..(last_index + m * (n - m) * (n - m))];
        last_index += m * (n - m) * (n - m);

        let s_buf = &mat_buf[last_index..(last_index + m * (n - m) * m)];

        let o_elems: Vec<F256Point> = o_buf.iter().map(|b| F256.make_point(*b)).collect();
        let p_elems: Vec<F256Point> = p_buf.iter().map(|b| F256.make_point(*b)).collect();
        let s_elems: Vec<F256Point> = s_buf.iter().map(|b| F256.make_point(*b)).collect();

        let o: Array2<F256Point> = Array2::from_shape_vec((n - m, m), o_elems)?;
        let p: Array3<F256Point> = Array3::from_shape_vec((m, n - m, n - m), p_elems)?;
        let s: Array3<F256Point> = Array3::from_shape_vec((m, n - m, m), s_elems)?;

        Ok(PrivateKey { n, m, seed, o_matrix: o, p_matrix_1: p, s_matrix: s })
    }
}

pub fn key_gen_seeded(
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
        let p3 = upper_triangular(&(&ot.dot(&p1).dot(&o) - &ot.dot(&p2))).unwrap();
        let ptop = ndarray::concatenate(ndarray::Axis(1), &[p1, p2]);
        let pbot = ndarray::concatenate(
            ndarray::Axis(1),
            &[Array2::zeros((m, n - m)).view(), p3.view()],
        );
        let pslice = ndarray::concatenate(
            ndarray::Axis(0),
            &[ptop.unwrap().view(), pbot.unwrap().view()],
        )
        .unwrap()
        .insert_axis(ndarray::Axis(0));
        p = ndarray::concatenate(ndarray::Axis(0), &[p.view(), pslice.view()]).unwrap();
        smats.push(&(&p1 + &p1.t()).dot(&o) + &p2);
    }

    let smat_views: Vec<ArrayView2<F256Point>> = smats.iter().map(|mat| mat.view()).collect();
    let s: Array3<F256Point> = stack(ndarray::Axis(0), &smat_views).unwrap();

    let esk = match PrivateKey::from_components(seed_sk, o, pmats.0, s, n, m) {
        Ok(k) => k,
        Err(err) => return Err("Error creating private key: \n".to_owned() + &err),
    };

    let epk = match PublicKey::from_p(p) {
        Ok(k) => k,
        Err(err) => return Err("Error creating public key: \n".to_owned() + &err),
    };

    Ok((esk, epk))
}

pub fn key_gen(n: usize, m: usize) -> Result<(PrivateKey, PublicKey), String> {
    let mut seed_sk = [0u8; SK_SEED_BYTES];
    let mut seed_pk = [0u8; PK_SEED_BYTES];
    OsRng.fill_bytes(&mut seed_sk);
    OsRng.fill_bytes(&mut seed_pk);
    return key_gen_seeded(seed_sk, seed_pk, n, m);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_public_key_from_components() {
        let seed_sk: [u8; SK_SEED_BYTES] = [0u8; SK_SEED_BYTES];
        let seed_pk: [u8; PK_SEED_BYTES] = [0u8; PK_SEED_BYTES];

        let o = expand_sk(&seed_sk, N_I, M_I);
        let ot = o.t();
        let pmats = expand_p(&seed_pk, N_I, M_I);
        let pk = PublicKey::from_components(&pmats.0, &pmats.1, &o, N_I, M_I).unwrap();

        for _round in 0..2 {
            for i in 0..M_I {
                let p1 = pk.p1(i);
                let p2 = pk.p2(i);

                assert_eq!(pmats.0.index_axis(ndarray::Axis(0), i), p1);
                assert_eq!(pmats.1.index_axis(ndarray::Axis(0), i), p2);

                let p3 = upper_triangular(&(&ot.dot(&p1).dot(&o) - &ot.dot(&p2))).unwrap();

                assert_eq!(p3, pk.p3(i));
            }
        }
    }

    #[test]
    fn test_public_key_from_elems() {
        let seed_sk: [u8; SK_SEED_BYTES] = [0u8; SK_SEED_BYTES];
        let seed_pk: [u8; PK_SEED_BYTES] = [0u8; PK_SEED_BYTES];

        let o: Array2<F256Point> = expand_sk(&seed_sk, N_I, M_I);
        let ot = o.t();
        let pmats = expand_p(&seed_pk, N_I, M_I);

        let mut p1_vec: Vec<ArrayView2<F256Point>> = Vec::new();
        let mut p2_vec: Vec<ArrayView2<F256Point>> = Vec::new();
        let mut p3_vec: Vec<Array2<F256Point>> = Vec::new();
        let mut elem_vec: Vec<F256Point> = Vec::new();

        for i in 0..M_I {
            let p1 = pmats.0.index_axis(ndarray::Axis(0), i);
            let p2 = pmats.1.index_axis(ndarray::Axis(0), i);
            let p3 = upper_triangular(&(&ot.dot(&p1).dot(&o) - &ot.dot(&p2))).unwrap();

            p1_vec.push(p1);
            p2_vec.push(p2);
            p3_vec.push(p3);
        }

        for row in 0..(N_I - M_I) {
            for col in row..(N_I - M_I) {
                for i in 0..M_I {
                    elem_vec.push(p1_vec.get(i).unwrap()[[row, col]]);
                }
            }
        }

        for row in 0..(N_I - M_I) {
            for col in 0..M_I {
                for i in 0..M_I {
                    elem_vec.push(p2_vec.get(i).unwrap()[[row, col]]);
                }
            }
        }

        for row in 0..M_I {
            for col in row..M_I {
                for i in 0..M_I {
                    elem_vec.push(p3_vec.get(i).unwrap()[[row, col]]);
                }
            }
        }

        let pk = PublicKey::from_elems(elem_vec, N_I, M_I).unwrap();

        for _round in 0..2 {
            for i in 0..M_I {
                let p1 = p1_vec.get(i).unwrap();
                let p2 = p2_vec.get(i).unwrap();
                let p3 = p3_vec.get(i).unwrap();

                assert_eq!(p1, pk.p1(i));
                assert_eq!(p2, pk.p2(i));
                assert_eq!(p3, pk.p3(i));
            }
        }
    }

    #[test]
    fn test_sign_verify() {
        let (sk, pk) = key_gen_seeded([0u8; SK_SEED_BYTES], [0u8; PK_SEED_BYTES], N_I, M_I).unwrap();
        let (mut sig, salt) = sk.sign(&[0u8; 64]).unwrap();
        assert!(pk.verify(&[0u8; 64], &sig.view(), &salt));
        sig[[12]] += F256.make_point(1);
        assert!(!pk.verify(&[0u8; 64], &sig.view(), &salt));
    }

    #[test]
    fn test_key_gen_seeded() {
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
        let pmats = expand_p(&seed_pk, N_I, M_I);
        let o = expand_sk(&seed_sk, N_I, M_I);
        let (_esk, epk) = key_gen_seeded(seed_sk, seed_pk, N_I, M_I).unwrap();
        for i in 0..M_I {
            let p1 = pmats.0.index_axis(ndarray::Axis(0), i);
            let p2 = pmats.1.index_axis(ndarray::Axis(0), i);
            let p3 = upper_triangular(&(&o.t().dot(&p1.dot(&o)) - &o.t().dot(&p2))).unwrap();
            assert_eq!(p1, epk.p1(i));
            assert_eq!(p2, epk.p2(i));
            assert_eq!(p3, epk.p3(i));
        }
    }

    #[test]
    fn test_public_key_serialize() {
        let (_esk, epk) = key_gen(N_I, M_I).unwrap();
        let mut buffer: Vec<u8> = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        epk.save(&mut cursor).unwrap();
        cursor.set_position(0);
        let restored = PublicKey::load(&mut cursor).unwrap();
        assert!(epk == restored);
    }

    #[test]
    fn test_private_key_serialize() {
        let (esk, _epk) = key_gen(N_I, M_I).unwrap();
        let mut buffer: Vec<u8> = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        esk.save(&mut cursor).unwrap();
        cursor.set_position(0);
        let restored = PrivateKey::load(&mut cursor).unwrap();
        assert!(esk == restored);
    }
}
