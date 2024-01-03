use core::iter::Sum;
use ndarray::ScalarOperand;
use num_traits::{Zero, One, Num, NumCast, ToPrimitive, Float};
use std::{ops::{Add, Mul, Sub, Div, Rem, Neg, AddAssign}, fmt::{UpperHex, LowerHex}, num::{ParseIntError, FpCategory}};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FiniteFieldU8<const M: u16>;

pub type F256Point = FiniteFieldValueU8<0b100011011>;
pub const F256: FiniteFieldU8<0b100011011> = FiniteFieldU8::<0b100011011>;

impl<const M: u16> FiniteFieldU8<{ M }> {
    pub fn make_point(self, value: u8) -> FiniteFieldValueU8<M> {
        FiniteFieldValueU8 { value }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct FiniteFieldValueU8<const F: u16> {
    value: u8,
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

impl<const M: u16> Div for FiniteFieldValueU8<{ M }> {
    type Output = Self;

    fn div(self, _rhs: Self) -> Self::Output {
        panic!("DIV OH NO");
    }
}


impl<const M: u16> Rem for FiniteFieldValueU8<{ M }> {
    type Output = Self;

    fn rem(self, _rhs: Self) -> Self::Output {
        todo!()
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


impl<const M: u16> Float for FiniteFieldValueU8<{ M }> {
    fn nan() -> Self {
        unimplemented!()
    }

    fn infinity() -> Self {
        unimplemented!()
    }

    fn neg_infinity() -> Self {
        unimplemented!()
    }

    fn neg_zero() -> Self {
        unimplemented!()
    }

    fn min_value() -> Self {
        Self::zero()
    }

    fn min_positive_value() -> Self {
        Self::zero()
    }

    fn max_value() -> Self {
        Self::zero() - Self::one()
    }

    fn is_nan(self) -> bool {
        false
    }

    fn is_infinite(self) -> bool {
        false
    }

    fn is_finite(self) -> bool {
        true
    }

    fn is_normal(self) -> bool {
        !self.is_zero()
    }

    fn classify(self) -> std::num::FpCategory {
        if self.is_normal() {
            FpCategory::Normal
        }
        else if self.is_zero() {
            FpCategory::Zero
        }
        else {
            FpCategory::Subnormal
        }
    }

    fn floor(self) -> Self {
        self
    }

    fn ceil(self) -> Self {
        self
    }

    fn round(self) -> Self {
        self
    }

    fn trunc(self) -> Self {
        self
    }

    fn fract(self) -> Self {
        Self::zero()
    }

    fn abs(self) -> Self {
        self
    }

    fn signum(self) -> Self {
        Self::one()
    }

    fn is_sign_positive(self) -> bool {
        true
    }

    fn is_sign_negative(self) -> bool {
        false
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        (self * a) + b
    }

    fn recip(self) -> Self {
        unimplemented!()
    }

    fn powi(self, n: i32) -> Self {
        let mut acc = Self::one();
        for _ in 0..n {
            acc = acc * self;
        }
        acc
    }

    fn powf(self, _n: Self) -> Self {
        unimplemented!()
    }

    fn sqrt(self) -> Self {
        unimplemented!()
    }

    fn exp(self) -> Self {
        unimplemented!()
    }

    fn exp2(self) -> Self {
        unimplemented!()
    }

    fn ln(self) -> Self {
        unimplemented!()
    }

    fn log(self, _base: Self) -> Self {
        unimplemented!()
    }

    fn log2(self) -> Self {
        unimplemented!()
    }

    fn log10(self) -> Self {
        unimplemented!()
    }

    fn max(self, other: Self) -> Self {
        if self.value > other.value {
            self
        }
        else {
            other
        }
    }

    fn min(self, other: Self) -> Self {
        if self.value > other.value {
            other
        }
        else {
            self
        }
    }

    fn abs_sub(self, other: Self) -> Self {
        self - other
    }

    fn cbrt(self) -> Self {
        unimplemented!()
    }

    fn hypot(self, _other: Self) -> Self {
        todo!()
    }

    fn sin(self) -> Self {
        unimplemented!()
    }

    fn cos(self) -> Self {
        unimplemented!()
    }

    fn tan(self) -> Self {
        unimplemented!()
    }

    fn asin(self) -> Self {
        unimplemented!()
    }

    fn acos(self) -> Self {
        unimplemented!()
    }

    fn atan(self) -> Self {
        unimplemented!()
    }

    fn atan2(self, _other: Self) -> Self {
        unimplemented!()
    }

    fn sin_cos(self) -> (Self, Self) {
        unimplemented!()
    }

    fn exp_m1(self) -> Self {
        unimplemented!()
    }

    fn ln_1p(self) -> Self {
        unimplemented!()
    }

    fn sinh(self) -> Self {
        unimplemented!()
    }

    fn cosh(self) -> Self {
        unimplemented!()
    }

    fn tanh(self) -> Self {
        unimplemented!()
    }

    fn asinh(self) -> Self {
        unimplemented!()
    }

    fn acosh(self) -> Self {
        unimplemented!()
    }

    fn atanh(self) -> Self {
        unimplemented!()
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        unimplemented!()
    }
}


impl<const M: u16> ScalarOperand for FiniteFieldValueU8<{ M }> {}


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
