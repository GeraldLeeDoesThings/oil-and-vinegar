use crate::field::{F256Point, F256};
use aes::{
    cipher::{generic_array::GenericArray, BlockEncrypt, KeyInit},
    Aes128,
};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

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

pub fn hash_concat(messages: &[&[u8]], m: usize) -> Vec<F256Point> {
    let mut hasher = Shake256::default();
    for message in messages {
        hasher.update(message);
    }
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
    let mut result: Vec<F256Point> = Vec::with_capacity(out);
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
        for byte in block.iter() {
            if bytes_remaining == 0 {
                break;
            }
            result.push(F256.make_point(*byte));
            bytes_remaining -= 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash() {
        let mut r = hash(&[0u8; 16], 0);
        assert_eq!(r.len(), 0);

        r = hash(&[0u8; 16], 1);
        assert_eq!(r.len(), 1);

        r = hash(&[0u8; 16], 16);
        assert_eq!(r.len(), 16);
    }

    #[test]
    fn test_hash_concat() {
        let mut r = hash_concat(&[&[0u8; 4], &[1u8; 4]], 0);
        let mut t = hash(&[0u8, 0u8, 0u8, 0u8, 1u8, 1u8, 1u8, 1u8], 0);
        assert_eq!(r.len(), 0);
        assert_eq!(r, t);

        r = hash_concat(&[&[0u8; 4], &[1u8; 4]], 1);
        t = hash(&[0u8, 0u8, 0u8, 0u8, 1u8, 1u8, 1u8, 1u8], 1);
        assert_eq!(r.len(), 1);
        assert_eq!(r, t);

        r = hash_concat(&[&[0u8; 4], &[1u8; 4]], 16);
        t = hash(&[0u8, 0u8, 0u8, 0u8, 1u8, 1u8, 1u8, 1u8], 16);
        assert_eq!(r.len(), 16);
        assert_eq!(r, t);
    }

    #[test]
    fn test_aes_hash() {
        let mut r = aes_hash(&[0u8; 16], 0);
        assert_eq!(r.len(), 0);

        r = aes_hash(&[0u8; 16], 1);
        assert_eq!(r.len(), 1);

        r = aes_hash(&[0u8; 16], 16);
        assert_eq!(r.len(), 16);

        r = aes_hash(&[0u8; 16], 17);
        assert_eq!(r.len(), 17);

        r = aes_hash(&[0u8; 16], 32);
        assert_eq!(r.len(), 32);

        r = aes_hash(&[0u8; 16], 40);
        assert_eq!(r.len(), 40);
    }
}
