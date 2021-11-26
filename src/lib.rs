pub mod presentation;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Input is malformed")]
    BadInput,
}
