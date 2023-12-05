use num_traits::Zero;
use std::ops::{Add, Mul};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FiniteFieldU8<const M: u16>;

pub type F256Point = FiniteFieldValueU8<0b100011011>;
pub const F256: FiniteFieldU8<0b100011011> = FiniteFieldU8::<0b100011011>;

impl<const M: u16> FiniteFieldU8<{ M }> {
    pub fn make_point(self, value: u8) -> FiniteFieldValueU8<M> {
        FiniteFieldValueU8 { value }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FiniteFieldValueU8<const F: u16> {
    value: u8,
}

impl<const M: u16> Zero for FiniteFieldValueU8<{ M }> {
    fn zero() -> Self {
        FiniteFieldValueU8 { value: 0 }
    }

    fn is_zero(&self) -> bool {
        self.value == 0
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const M: u16> Add for FiniteFieldValueU8<{ M }> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self.value ^ rhs.value,
        }
    }
}

impl<const M: u16> Mul for FiniteFieldValueU8<{ M }> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut accumulator: u16 = 0;
        let mut lhsv = self.value as u16;
        let mut rhsv = rhs.value as u16;
        for _ in 0..8 {
            accumulator ^= -((rhsv & 1) as i16) as u16 & lhsv;
            let mask = -(((lhsv >> (14 - M.leading_zeros())) & 1) as i16) as u16;
            lhsv = (lhsv << 1) ^ (M & mask);
            rhsv >>= 1;
        }
        Self {
            value: accumulator as u8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let f = FiniteFieldU8::<0b111>;
        let p0 = f.make_point(0);
        let p1 = f.make_point(1);
        let p2 = f.make_point(2);
        let p3 = f.make_point(3);
        assert_eq!(p0 + p1, p1);
        assert_eq!(p1 + p2, p3);
        assert_eq!(p2 + p3, p1);
        assert_eq!(p3 + p3, p0);
    }

    #[test]
    fn test_mul() {
        let f1 = FiniteFieldU8::<0b111>;
        let p0 = f1.make_point(0);
        let p1 = f1.make_point(1);
        let p2 = f1.make_point(2);
        let p3 = f1.make_point(3);
        assert_eq!(p0 * p1, p0);
        assert_eq!(p1 * p2, p2);
        assert_eq!(p2 * p3, p1);
        assert_eq!(p2 * p2, p3);

        let f2 = FiniteFieldU8::<0b100011011>;
        assert_eq!(
            f2.make_point(0b1010011) * f2.make_point(0b11001010),
            f2.make_point(1)
        );
    }
}
