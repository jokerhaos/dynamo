// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preserve wire error categories and causes when a Python router forwards a stream.
use dynamo_runtime::error::DynamoError;
use pyo3::prelude::*;

// Retain ValueError compatibility for callers of AsyncResponseStream. A distinct
// class prevents arbitrary Python ValueErrors from being reclassified by attributes.
pyo3::create_exception!(
    dynamo._core,
    ForwardedDynamoError,
    pyo3::exceptions::PyValueError
);

pub fn dynamo_error_to_pyerr(error: DynamoError) -> PyErr {
    let exception = ForwardedDynamoError::new_err(error.to_string());
    let attributes = Python::with_gil(|py| -> PyResult<()> {
        let payload = serde_json::to_string(&error)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        exception.value(py).setattr("_dynamo_error", payload)
    });
    match attributes {
        Ok(()) => exception,
        Err(error) => error,
    }
}

pub fn restore_forwarded_error(py: Python<'_>, error: &PyErr) -> Option<DynamoError> {
    if !error.is_instance_of::<ForwardedDynamoError>(py) {
        return None;
    }
    let payload = error
        .value(py)
        .getattr("_dynamo_error")
        .ok()?
        .extract::<String>()
        .ok()?;
    serde_json::from_str(&payload).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::error::{BackendError, ErrorType};

    #[test]
    fn preserves_categories_messages_and_causes_across_multiple_python_hops() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            for category in [
                ErrorType::ResourceExhausted,
                ErrorType::WorkerOverloaded,
                ErrorType::Unavailable,
                ErrorType::Cancelled,
                ErrorType::Backend(BackendError::InvalidArgument),
                ErrorType::Backend(BackendError::Unknown),
            ] {
                let original = DynamoError::builder()
                    .error_type(category)
                    .message(r#"{"code":429,"message":"test rejection"}"#)
                    .cause(DynamoError::msg("underlying cause"))
                    .build();
                let expected = serde_json::to_value(&original).unwrap();
                let mut forwarded = original;
                for _ in 0..3 {
                    let exception = dynamo_error_to_pyerr(forwarded);
                    assert!(exception.is_instance_of::<pyo3::exceptions::PyValueError>(py));
                    forwarded = restore_forwarded_error(py, &exception).unwrap();
                    assert_eq!(serde_json::to_value(&forwarded).unwrap(), expected);
                }
            }
        });
    }

    #[test]
    fn ordinary_or_malformed_exceptions_do_not_restore_a_category() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let ordinary = pyo3::exceptions::PyValueError::new_err("invalid input");
            ordinary.value(py).setattr("_dynamo_error", "{}").unwrap();
            assert!(restore_forwarded_error(py, &ordinary).is_none());
            let malformed = ForwardedDynamoError::new_err("invalid payload");
            assert!(restore_forwarded_error(py, &malformed).is_none());
            malformed
                .value(py)
                .setattr("_dynamo_error", "not JSON")
                .unwrap();
            assert!(restore_forwarded_error(py, &malformed).is_none());
        });
    }
}
