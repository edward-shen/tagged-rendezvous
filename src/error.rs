use std::error::Error;
use std::fmt::Display;

use crate::NodeId;

/// A duplicate id was provided. This contains the duplicated ID that was
/// provided.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct DuplicateIdError(pub(crate) NodeId);

#[cfg(not(tarpaulin_include))]
impl Display for DuplicateIdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A duplicate ID {} was provided", self.0)
    }
}

impl Error for DuplicateIdError {}
