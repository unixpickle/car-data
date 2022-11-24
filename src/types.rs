use std::convert::Infallible;
use std::fmt::Display;
use std::str::FromStr;

use rusqlite::types::{FromSql, FromSqlError, ToSqlOutput, ValueRef};
use rusqlite::ToSql;

#[derive(Clone, Debug, Default)]
pub enum PriceUnit {
    #[default]
    Cents,
}

#[derive(Clone, Debug)]
pub struct Price {
    pub value: u64,
    pub unit: PriceUnit,
}

impl FromStr for Price {
    type Err = <f64 as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut processed = s.trim().replace(",", "");
        let mut unit = PriceUnit::default();
        if processed.starts_with("$") {
            unit = PriceUnit::Cents;
            processed = processed.replace("$", "");
        }
        Ok(Price {
            value: (f64::from_str(&processed)? * 100.0).round() as u64,
            unit: unit,
        })
    }
}

impl Display for Price {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.unit {
            PriceUnit::Cents => {
                write!(f, "${:.02}", ((self.value as f64) / 100.0))?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub enum DistanceUnit {
    #[default]
    Miles,
}

#[derive(Clone, Debug)]
pub struct Distance {
    pub value: u64,
    pub unit: DistanceUnit,
}

impl FromStr for Distance {
    type Err = <f64 as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut processed = s.trim().replace(",", "");
        let mut unit = DistanceUnit::default();
        if processed.ends_with(" mi") {
            unit = DistanceUnit::Miles;
            processed = processed.replace(" mi", "");
        }
        Ok(Distance {
            value: (f64::from_str(&processed)?).round() as u64,
            unit: unit,
        })
    }
}

impl Display for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.unit {
            DistanceUnit::Miles => {
                write!(f, "{} mi", self.value)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum DriveType {
    TwoWheelFront,
    TwoWheelRear,
    FourWheel,
    Other(String),
}

impl FromStr for DriveType {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "FWD" => DriveType::TwoWheelFront,
            "RWD" => DriveType::TwoWheelRear,
            "AWD4WD" => DriveType::FourWheel,
            x => DriveType::Other(x.to_owned()),
        })
    }
}

impl Display for DriveType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TwoWheelFront => write!(f, "FWD")?,
            Self::TwoWheelRear => write!(f, "RWD")?,
            Self::FourWheel => write!(f, "AWD4WD")?,
            Self::Other(x) => write!(f, "{}", x)?,
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum FuelType {
    Gasoline,
    Hybrid,
    Diesel,
    Electric,
    Hydrogen,
    Alternative,
}

impl FromStr for FuelType {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "Gasoline" => FuelType::Gasoline,
            "Hybrid" => FuelType::Hybrid,
            "Diesel" => FuelType::Diesel,
            "Electric" => FuelType::Electric,
            "Hydrogen" => FuelType::Hydrogen,
            _ => FuelType::Alternative,
        })
    }
}

impl Display for FuelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                FuelType::Gasoline => "Gasoline",
                FuelType::Hybrid => "Hybrid",
                FuelType::Diesel => "Diesel",
                FuelType::Electric => "Electric",
                FuelType::Hydrogen => "Hydrogen",
                FuelType::Alternative => "Alternative",
            }
        )
    }
}

#[derive(Clone, Debug)]
pub struct OwnerInfo {
    pub id: String,
    pub name: Option<String>,
    pub website: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Listing {
    pub website: String,
    pub website_id: String,
    pub title: String,
    pub price: Option<Price>,
    pub make: Option<String>,
    pub model: Option<String>,
    pub year: Option<u64>,
    pub odometer: Option<Distance>,
    pub engine_description: Option<String>,
    pub exterior_color: Option<String>,
    pub interior_color: Option<String>,
    pub drive_type: Option<DriveType>,
    pub fuel_type: Option<FuelType>,
    pub fuel_economy: Option<Vec<String>>,
    pub owners: Option<Vec<OwnerInfo>>,
    pub vin: Option<String>,
    pub stock_number: Option<String>,
    pub comments: Option<String>,
    pub image_urls: Option<Vec<String>>,
}

macro_rules! sql_string_obj {
    ($data_type:ty) => {
        impl ToSql for $data_type {
            fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
                Ok(ToSqlOutput::Owned(rusqlite::types::Value::Text(format!(
                    "{}",
                    self
                ))))
            }
        }

        impl FromSql for $data_type {
            fn column_result(
                value: rusqlite::types::ValueRef<'_>,
            ) -> rusqlite::types::FromSqlResult<Self> {
                match value {
                    ValueRef::Text(x) => String::from_utf8(Vec::from(x))
                        .map_err(|x| FromSqlError::Other(Box::new(x)))?
                        .parse()
                        .map_err(|x| FromSqlError::Other(Box::new(x))),
                    _ => Err(FromSqlError::InvalidType),
                }
            }
        }
    };
}

sql_string_obj!(Price);
sql_string_obj!(Distance);
sql_string_obj!(DriveType);
sql_string_obj!(FuelType);
