use std::str::FromStr;
use std::{collections::HashMap, future::Future, pin::Pin};

use crate::parse_util::{inner_text, FromJSON};
use reqwest::{RequestBuilder, Response};
use scraper::{Html, Selector};
use serde_json::Value;

const VERSION: &str = env!("CARGO_PKG_VERSION");

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

#[derive(Clone, Debug, Hash)]
pub enum FeatureCategory {
    Exterior,
    Interior,
    Safety,
    Mechanical,
    Technology,
    Other,
}

#[derive(Clone, Debug)]
pub enum DriveType {
    TwoWheel,
    FourWheel,
    Other(String),
}

#[derive(Clone, Debug)]
pub struct OwnerInfo {
    pub id: String,
    pub name: Option<String>,
    pub website: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Listing {
    pub title: String,
    pub price: Option<Price>,
    pub make: Option<String>,
    pub model: Option<String>,
    pub year: Option<u64>,
    pub odometer: Option<Distance>,
    pub engine_description: Option<String>,
    pub color: Option<String>,
    pub drive_type: Option<DriveType>,
    pub fuel_economy: Option<Vec<String>>,
    pub owners: Option<Vec<OwnerInfo>>,
    pub vin: Option<String>,
    pub stock_number: Option<String>,
    pub comments: Option<String>,
    pub features: HashMap<FeatureCategory, Vec<String>>,
}

pub struct Client {
    client: reqwest::Client,
    num_retries: i32,
}

impl Client {
    pub fn new(num_retries: i32) -> Client {
        Client {
            client: reqwest::Client::new(),
            num_retries: num_retries,
        }
    }

    pub async fn run<R: Request>(&mut self, req: R) -> anyhow::Result<R::Output> {
        let mut last_err: anyhow::Error = anyhow::Error::msg("UNREACHABLE");
        for _ in 0..self.num_retries {
            let builder = req
                .build_request(&self)
                .header("host", "www.kbb.com")
                .header("user-agent", format!("cardata/{}", VERSION));
            let result = builder.send().await;
            match result {
                Err(e) => {
                    last_err = e.into();
                }
                Ok(resp) => {
                    let output = req.handle_response(resp).await;
                    match output {
                        Err(e) => {
                            last_err = e.into();
                        }
                        Ok(x) => {
                            return Ok(x);
                        }
                    }
                }
            };
        }
        Err(last_err)
    }
}

pub trait Request {
    type Output;
    type Err: Into<anyhow::Error>;

    fn build_request(&self, client: &Client) -> RequestBuilder;

    fn handle_response(
        &self,
        resp: Response,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Output, Self::Err>>>>;
}

// A request for fetching information about an individual car listing.
pub struct ListingRequest(pub String);

impl Request for ListingRequest {
    type Output = Option<Listing>;
    type Err = anyhow::Error;

    fn build_request(&self, client: &Client) -> RequestBuilder {
        client.client.get(format!(
            "https://www.kbb.com/cars-for-sale/vehicledetails.xhtml?listingId={}",
            self.0
        ))
    }

    fn handle_response(
        &self,
        resp: Response,
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<Self::Output>>>> {
        Box::pin(async {
            let text = resp.text().await?;
            let doc = Html::parse_fragment(&text);
            let titles: Vec<_> = doc.select(&Selector::parse("h1").unwrap()).collect();
            if titles.len() == 0 {
                // The "car no longer available" page.
                return Ok(None);
            } else if titles.len() != 1 {
                return Err(anyhow::Error::msg("no title heading found on listing page"));
            }

            let doc_info = extract_doc_json(&text)?;

            Ok(Some(Listing {
                title: inner_text(&titles[0]),
                price: {
                    f64::extract_from_json(
                        &doc_info,
                        "initialState.birf.pageData.page.vehicle.price",
                    )
                    .ok()
                    .map(|x| Price {
                        value: 100 * (x as u64),
                        unit: PriceUnit::Cents,
                    })
                },
                make: {
                    <Vec<String>>::extract_from_json(
                        &doc_info,
                        "initialState.birf.pageData.page.vehicle.makeName",
                    )
                    .ok()
                    .and_then(vec_into_first)
                },
                model: {
                    <Vec<String>>::extract_from_json(
                        &doc_info,
                        "initialState.birf.pageData.page.vehicle.modelName",
                    )
                    .ok()
                    .and_then(vec_into_first)
                },
                year: {
                    u64::extract_from_json(
                        &doc_info,
                        "initialState.birf.pageData.page.vehicle.car_year",
                    )
                    .ok()
                },
                odometer: {
                    String::extract_from_json(
                        &doc_info,
                        "initialState.birf.pageData.page.vehicle.odometer",
                    )
                    .ok()
                    .and_then(|x| Distance::from_str(&x).ok())
                },
                engine_description: None,
                color: {
                    <Vec<String>>::extract_from_json(
                        &doc_info,
                        "initialState.birf.pageData.page.vehicle.color",
                    )
                    .ok()
                    .and_then(vec_into_first)
                },
                drive_type: None,
                fuel_economy: {
                    <Vec<String>>::extract_from_json(
                        &doc_info,
                        "initialState.birf.pageData.page.vehicle.fuelEconomy",
                    )
                    .ok()
                },
                owners: {
                    <HashMap<String, Value>>::extract_from_json(&doc_info, "initialState.owners")
                        .ok()
                        .map(|x| {
                            let mut result = Vec::new();
                            for (owner_id, owner_info) in x.into_iter() {
                                let name = String::extract_from_json(&owner_info, "name").ok();
                                let website =
                                    String::extract_from_json(&owner_info, "website.href").ok();
                                result.push(OwnerInfo {
                                    id: owner_id,
                                    name,
                                    website,
                                });
                            }
                            result
                        })
                },
                vin: String::extract_from_json(
                    &doc_info,
                    "initialState.birf.pageData.page.vehicle.vin",
                )
                .ok(),
                stock_number: String::extract_from_json(
                    &doc_info,
                    "initialState.birf.pageData.page.vehicle.stockNumber",
                )
                .ok(),
                comments: None,
                features: HashMap::new(),
            }))
        })
    }
}

fn extract_doc_json(contents: &str) -> anyhow::Result<serde_json::Value> {
    let preamble = "window.__BONNET_DATA__=";
    if let Some(index) = contents.find(preamble) {
        let mut suffix = &contents[index + preamble.len()..];
        if let Some(index) = suffix.find("</script>") {
            suffix = &suffix[..index];
        }
        Ok(serde_json::from_str(suffix)?)
    } else {
        Err(anyhow::Error::msg("could not find JSON data in document"))
    }
}

fn vec_into_first<T>(list: Vec<T>) -> Option<T> {
    for x in list {
        return Some(x);
    }
    None
}
