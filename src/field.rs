use std::ops::{Add, Mul};

#[derive(Debug, Copy, Clone, PartialEq)]
struct FiniteField {
    mod_polynomial: u64,
}

impl FiniteField {
    pub fn make_point(self, value: u64) -> FiniteFieldValue {
        FiniteFieldValue { field: self, value }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct FiniteFieldValue {
    field: FiniteField,
    value: u64,
}

impl Add for FiniteFieldValue {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.field, rhs.field,
            "Mismatched fields ({:#b} and {:#b}) when computing {:#b} + {:#b}",
            self.field.mod_polynomial, rhs.field.mod_polynomial, self.value, rhs.value
        );
        Self {
            field: self.field,
            value: self.value ^ rhs.value,
        }
    }
}

impl Mul for FiniteFieldValue {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.field, rhs.field,
            "Mismatched fields ({:#b} and {:#b}) when computing {:#b} * {:#b}",
            self.field.mod_polynomial, rhs.field.mod_polynomial, self.value, rhs.value
        );
        let mut accumulator: u64 = 0;
        let mut lhsv = self.value;
        let mut rhsv = rhs.value;
        let mod_polynomial = self.field.mod_polynomial;
        for _ in 0..64 {
            accumulator ^= -((rhsv & 1) as i64) as u64 & lhsv;
            let mask = -(((lhsv >> 62 - mod_polynomial.leading_zeros()) & 1) as i64) as u64;
            lhsv = (lhsv << 1) ^ (mod_polynomial & mask);
            rhsv >>= 1;
        }
        Self {
            field: self.field,
            value: accumulator,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::field::FiniteField;

    #[test]
    fn test_add() {
        let f = FiniteField {
            mod_polynomial: 0b111,
        };
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
    #[should_panic(expected = "Mismatched fields (0b111 and 0b1001001) when computing 0b10 + 0b11")]
    fn test_add_fail() {
        let f1 = FiniteField {
            mod_polynomial: 0b111,
        };
        let f2 = FiniteField {
            mod_polynomial: 0b1001001,
        };
        let _ = f1.make_point(2) + f2.make_point(3);
    }

    #[test]
    fn test_mul() {
        let f1 = FiniteField {
            mod_polynomial: 0b111,
        };
        let p0 = f1.make_point(0);
        let p1 = f1.make_point(1);
        let p2 = f1.make_point(2);
        let p3 = f1.make_point(3);
        assert_eq!(p0 * p1, p0);
        assert_eq!(p1 * p2, p2);
        assert_eq!(p2 * p3, p1);
        assert_eq!(p2 * p2, p3);

        let f2 = FiniteField {
            mod_polynomial: 0b100011011,
        };
        assert_eq!(
            f2.make_point(0b1010011) * f2.make_point(0b11001010),
            f2.make_point(1)
        );
    }

    #[test]
    #[should_panic(expected = "Mismatched fields (0b111 and 0b1001001) when computing 0b10 * 0b11")]
    fn test_mul_fail() {
        let f1 = FiniteField {
            mod_polynomial: 0b111,
        };
        let f2 = FiniteField {
            mod_polynomial: 0b1001001,
        };
        let _ = f1.make_point(2) * f2.make_point(3);
    }
}
