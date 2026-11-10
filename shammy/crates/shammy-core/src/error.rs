use std::fmt;

/// Everything that can go wrong, with a message a person can act on.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// The file is not a Shammy file, or is damaged.
    Format(String),
    /// A checksum did not match: the named part of the file is corrupt.
    Corrupt(String),
    /// The caller passed something the file cannot accept.
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Format(m) | Error::Corrupt(m) | Error::Invalid(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
