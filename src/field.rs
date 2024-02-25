use core::{
    fmt,
    iter::{Product, Sum},
};
use ndarray::ScalarOperand;
use num_traits::{FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize, Serializer};
use std::{
    fmt::{Debug, Display, LowerExp, LowerHex, UpperExp, UpperHex},
    num::ParseIntError,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FiniteFieldU8<const M: u16>;

pub type F256Point = FiniteFieldValueU8<0b100011011>;
pub const F256: FiniteFieldU8<0b100011011> = FiniteFieldU8::<0b100011011>;

impl<const M: u16> FiniteFieldU8<{ M }> {
    pub fn make_point(self, value: u8) -> FiniteFieldValueU8<M> {
        FiniteFieldValueU8 { value }
    }
}

#[derive(Copy, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct FiniteFieldValueU8<const F: u16> {
    value: u8,
}

impl<const M: u16> FiniteFieldValueU8<{ M }> {
    pub fn to_byte(&self) -> u8 {
        self.value
    }
}

impl<const M: u16> Debug for FiniteFieldValueU8<{ M }> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.serialize_u8(self.value)
    }
}

impl<const M: u16> UpperHex for FiniteFieldValueU8<{ M }> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        UpperHex::fmt(&self.value, f)
    }
}

impl<const M: u16> LowerHex for FiniteFieldValueU8<{ M }> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        LowerHex::fmt(&self.value, f)
    }
}

impl<const M: u16> Zero for FiniteFieldValueU8<{ M }> {
    fn zero() -> Self {
        FiniteFieldValueU8 { value: 0 }
    }

    fn is_zero(&self) -> bool {
        self.value == 0
    }
}

impl<const M: u16> One for FiniteFieldValueU8<{ M }> {
    fn one() -> Self {
        FiniteFieldValueU8 { value: 1 }
    }

    fn is_one(&self) -> bool {
        self.value == 1
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

#[allow(clippy::suspicious_arithmetic_impl)]
impl<const M: u16> Sub for FiniteFieldValueU8<{ M }> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
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

#[derive(Debug, PartialEq)]
struct DivResult<T> {
    quotient: T,
    remainder: T,
}

fn most_sig_one(v: u16) -> Option<u16> {
    let mut best = 0u16;
    let mut check = 1u16;
    let mut shift = 0u16;
    for check_shift in 0..u16::BITS {
        if v & check > 0 {
            best = check;
            shift = check_shift as u16;
        }
        check <<= 1;
    }
    if best == 0 {
        None
    } else {
        Some(shift)
    }
}

fn poly_div(lhs: u16, rhs: u16) -> Result<DivResult<u16>, String> {
    let mut r: u16 = 0;
    let rhs_mso = most_sig_one(rhs).ok_or("Division by zero!")?;

    if lhs == 0 && rhs != 0 {
        return Ok(DivResult {
            quotient: 0,
            remainder: 0,
        });
    }

    let lhs_mso = most_sig_one(lhs).unwrap();

    if rhs_mso > lhs_mso || rhs > lhs {
        return Ok(DivResult {
            quotient: 0,
            remainder: rhs,
        });
    }

    let mut dividend = lhs;

    for shift in (0..=(lhs_mso - rhs_mso)).rev() {
        let adj = rhs << shift;
        let mask = 1u16 << (rhs_mso + shift);
        if dividend & mask > 0 {
            dividend = dividend ^ adj;
            r += 1 << shift;
        }
    }

    Ok(DivResult {
        quotient: r,
        remainder: dividend,
    })
}

impl<const M: u16> FiniteFieldValueU8<{ M }> {
    fn inverse(&self) -> Option<FiniteFieldValueU8<M>> {
        let mut t = FiniteFieldValueU8::<M>::zero();
        let mut r = FiniteFieldValueU8::<M>::zero(); // M;
        let mut new_t = FiniteFieldValueU8::<M>::one(); // 1;
        let mut new_r = FiniteFieldValueU8::<M> { value: self.value }; // self.value as u16;

        if new_r.value != 0 {
            let quotient = FiniteFieldValueU8::<M> {
                value: poly_div(M, new_r.value as u16).ok()?.quotient as u8,
            };
            (r, new_r) = (new_r, r - quotient * new_r);
            (t, new_t) = (new_t, t - quotient * new_t);
        }

        while new_r.value != 0 {
            let quotient = FiniteFieldValueU8::<M> {
                value: poly_div(r.value as u16, new_r.value as u16).ok()?.quotient as u8,
            };
            (r, new_r) = (new_r, r - quotient * new_r);
            (t, new_t) = (new_t, t - quotient * new_t);
        }

        if r.value > 1 {
            None
        } else if r.value == 1 {
            Some(t)
        } else {
            panic!("Division by zero?")
        }
    }
}

impl<const M: u16> Div for FiniteFieldValueU8<{ M }> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse().unwrap()
    }
}

impl<const M: u16> Rem for FiniteFieldValueU8<{ M }> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            panic!("Divide by zero!")
        } else {
            Self::Output::zero()
        }
    }
}

impl<const M: u16> Num for FiniteFieldValueU8<{ M }> {
    type FromStrRadixErr = ParseIntError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let value = u8::from_str_radix(str, radix)?;

        Ok(FiniteFieldValueU8 { value })
    }
}

impl<const M: u16> ToPrimitive for FiniteFieldValueU8<{ M }> {
    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
}

impl<const M: u16> NumCast for FiniteFieldValueU8<{ M }> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        let value = n.to_u8()?;
        Some(FiniteFieldValueU8 { value })
    }
}

impl<const M: u16> Neg for FiniteFieldValueU8<{ M }> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self
    }
}

impl<const M: u16> Signed for FiniteFieldValueU8<{ M }> {
    fn abs(&self) -> Self {
        self.clone()
    }

    fn abs_sub(&self, other: &Self) -> Self {
        *self - *other
    }

    fn signum(&self) -> Self {
        Self::one()
    }

    fn is_positive(&self) -> bool {
        !Self::is_zero(self)
    }

    fn is_negative(&self) -> bool {
        false
    }
}

impl<const M: u16> AddAssign for FiniteFieldValueU8<{ M }> {
    fn add_assign(&mut self, rhs: Self) {
        let result = *self + rhs;
        self.value = result.value;
    }
}

impl<const M: u16> Sum for FiniteFieldValueU8<{ M }> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut acc = Self::zero();
        for v in iter {
            acc += v;
        }
        acc
    }
}

impl<const M: u16> SubAssign for FiniteFieldValueU8<{ M }> {
    fn sub_assign(&mut self, rhs: Self) {
        self.value = Self::sub(*self, rhs).value;
    }
}

impl<const M: u16> MulAssign for FiniteFieldValueU8<{ M }> {
    fn mul_assign(&mut self, rhs: Self) {
        self.value = Self::mul(*self, rhs).value;
    }
}

impl<const M: u16> DivAssign for FiniteFieldValueU8<{ M }> {
    fn div_assign(&mut self, rhs: Self) {
        self.value = Self::div(*self, rhs).value;
    }
}

impl<const M: u16> RemAssign for FiniteFieldValueU8<{ M }> {
    fn rem_assign(&mut self, rhs: Self) {
        self.value = Self::rem(*self, rhs).value;
    }
}

impl<const M: u16> UpperExp for FiniteFieldValueU8<{ M }> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&self.value, f)
    }
}

impl<const M: u16> LowerExp for FiniteFieldValueU8<{ M }> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerExp::fmt(&self.value, f)
    }
}

impl<const M: u16> FromPrimitive for FiniteFieldValueU8<{ M }> {
    fn from_i64(n: i64) -> Option<Self> {
        return Some(Self { value: n as u8 });
    }

    fn from_u64(n: u64) -> Option<Self> {
        return Some(Self { value: n as u8 });
    }
}

impl<const M: u16> Display for FiniteFieldValueU8<{ M }> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{:x?}", self.value))
    }
}

impl<const M: u16> Product for FiniteFieldValueU8<{ M }> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut acc = Self::one();
        for v in iter {
            acc *= v;
        }
        acc
    }
}

impl<const M: u16> ScalarOperand for FiniteFieldValueU8<{ M }> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_most_sig_one() {
        assert_eq!(most_sig_one(0b0), None);
        assert_eq!(most_sig_one(0b1), Some(0));
        assert_eq!(most_sig_one(0b10), Some(1));
        assert_eq!(most_sig_one(0b101101), Some(5));
    }

    #[test]
    fn test_poly_div() {
        assert_eq!(poly_div(0, 0), Err("Division by zero!".to_string()));
        assert_eq!(
            poly_div(0, 1),
            Ok(DivResult {
                quotient: 0,
                remainder: 0
            })
        );
        assert_eq!(
            poly_div(1, 1),
            Ok(DivResult {
                quotient: 1,
                remainder: 0
            })
        );
        assert_eq!(
            poly_div(0b11, 1),
            Ok(DivResult {
                quotient: 0b11,
                remainder: 0
            })
        );
        assert_eq!(
            poly_div(0b11, 0b11),
            Ok(DivResult {
                quotient: 1,
                remainder: 0
            })
        );
        assert_eq!(
            poly_div(0b11, 0b10),
            Ok(DivResult {
                quotient: 1,
                remainder: 1
            })
        );
        assert_eq!(
            poly_div(0b101, 0b11),
            Ok(DivResult {
                quotient: 0b11,
                remainder: 0
            })
        );
        assert_eq!(
            poly_div(0b111, 0b11),
            Ok(DivResult {
                quotient: 0b10,
                remainder: 1
            })
        );
        assert_eq!(
            poly_div(0b1100011, 0b11),
            Ok(DivResult {
                quotient: 0b100001,
                remainder: 0
            })
        );
        assert_eq!(
            poly_div(0b1100011, 0b11),
            Ok(DivResult {
                quotient: 0b100001,
                remainder: 0
            })
        );
    }

    #[test]
    fn test_inverse() {
        let one = F256Point::one();
        let two = F256Point { value: 0b10 };
        let a = F256Point { value: 0b1101 };
        let b = F256Point { value: 0b11011101 };
        let c = F256Point { value: 0b11111111 };
        assert_eq!(one, one * one.inverse().unwrap());
        assert_eq!(one, two * two.inverse().unwrap());
        assert_eq!(one, a * a.inverse().unwrap());
        assert_eq!(one, b * b.inverse().unwrap());
        assert_eq!(one, c * c.inverse().unwrap());
    }

    #[test]
    fn test_div() {
        fn test_pair<const M: u16>(lhs: FiniteFieldValueU8<M>, rhs: FiniteFieldValueU8<M>) {
            let result = lhs / rhs;
            assert_eq!(lhs, rhs * result);
        }

        let zero = F256Point::zero();
        let one = F256Point::one();
        let two = F256Point { value: 0b10 };
        let a = F256Point { value: 0b1101 };
        let b = F256Point { value: 0b11011101 };
        let c = F256Point { value: 0b11111111 };

        test_pair(zero, two);
        test_pair(one, two);
        test_pair(two, one);
        test_pair(c, b);
        test_pair(c, a);
        test_pair(c, c);
    }

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
