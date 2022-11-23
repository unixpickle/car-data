use scraper::ElementRef;
use serde_json::Value;
use std::{collections::HashMap, fmt::Write};

pub fn inner_text(obj: &ElementRef) -> String {
    let mut result = String::new();
    for x in obj.text() {
        write!(&mut result, "{} ", x).unwrap();
    }
    result.trim().to_owned()
}

pub trait FromJSON
where
    Self: Sized,
{
    fn from_json(value: &Value) -> anyhow::Result<Self>;

    fn extract_from_json(root: &Value, path: &str) -> anyhow::Result<Self> {
        let mut cur_obj = root;
        for part in path.split(".") {
            if let Value::Object(obj) = cur_obj {
                if let Some(x) = obj.get(part) {
                    cur_obj = x;
                } else {
                    return Err(anyhow::Error::msg(format!(
                        "object path not found: {}",
                        path
                    )));
                }
            } else {
                return Err(anyhow::Error::msg(format!(
                    "incorrect type in object path: {}",
                    path
                )));
            }
        }
        match Self::from_json(cur_obj) {
            Ok(x) => Ok(x),
            Err(e) => Err(anyhow::Error::msg(format!(
                "error for object path {}: {}",
                path, e
            ))),
            other => other,
        }
    }
}

impl FromJSON for Value {
    fn from_json(value: &Value) -> anyhow::Result<Self> {
        Ok(value.clone())
    }
}

impl FromJSON for f64 {
    fn from_json(value: &Value) -> anyhow::Result<Self> {
        match value {
            Value::Number(x) => {
                if let Some(f) = x.as_f64() {
                    Ok(f)
                } else {
                    Err(anyhow::Error::msg(format!("{} is not an f64", x)))
                }
            }
            _ => Err(anyhow::Error::msg(format!("{} is not a number", value))),
        }
    }
}

impl FromJSON for u64 {
    fn from_json(value: &Value) -> anyhow::Result<Self> {
        match value {
            Value::Number(x) => {
                if let Some(f) = x.as_u64() {
                    Ok(f)
                } else {
                    Err(anyhow::Error::msg(format!("{} is not a u64", x)))
                }
            }
            _ => Err(anyhow::Error::msg(format!("{} is not a number", value))),
        }
    }
}

impl FromJSON for String {
    fn from_json(value: &Value) -> anyhow::Result<Self> {
        match value {
            Value::String(x) => Ok(x.clone()),
            _ => Err(anyhow::Error::msg(format!("{} is not a string", value))),
        }
    }
}

impl<T: FromJSON> FromJSON for Vec<T> {
    fn from_json(value: &Value) -> anyhow::Result<Self> {
        match value {
            Value::Array(x) => x
                .iter()
                .map(|x| T::from_json(x))
                .collect::<anyhow::Result<Vec<T>>>(),
            _ => Err(anyhow::Error::msg(format!("{} is not an array", value))),
        }
    }
}

impl<T: FromJSON> FromJSON for HashMap<String, T> {
    fn from_json(value: &Value) -> anyhow::Result<Self> {
        match value {
            Value::Object(x) => x
                .iter()
                .map(|(k, v)| T::from_json(v).map(|x| (k.clone(), x)))
                .collect::<anyhow::Result<HashMap<String, T>>>(),
            _ => Err(anyhow::Error::msg(format!("{} is not an object", value))),
        }
    }
}
