use std::fmt;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

#[cfg(test)]
mod tests {
    use crate::nvidia::ComputeCapability;

    #[test]
    fn cc_ordering() {
        let cc_7_0 = ComputeCapability::new(7, 0);
        let cc_7_5 = ComputeCapability::new(7, 5);
        let cc_9_0 = ComputeCapability::new(9, 0);
        let cc_12_0 = ComputeCapability::new(12, 0);

        assert!(cc_7_0 < cc_7_5);
        assert!(cc_7_5 < cc_9_0);
        assert!(cc_9_0 < cc_12_0);
        assert_eq!(cc_9_0, ComputeCapability::new(9, 0));
    }

    #[test]
    fn cc_display() {
        assert_eq!(ComputeCapability::new(9, 0).to_string(), "9.0");
        assert_eq!(ComputeCapability::new(12, 3).to_string(), "12.3");
    }
}
