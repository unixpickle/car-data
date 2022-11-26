use crate::types::{Listing, OwnerInfo, Price, PriceUnit};
use std::{collections::HashMap, future::Future, path::PathBuf, pin::Pin, time::Duration};

use crate::parse_util::{inner_text, FromJSON};
use reqwest::{RequestBuilder, Response};
use scraper::{Html, Selector};
use serde_json::Value;
use tokio::{fs::File, io::AsyncWriteExt, time::sleep};

const VERSION: &str = env!("CARGO_PKG_VERSION");

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
        for i in 0..self.num_retries {
            let builder = req
                .build_request(&self)
                .timeout(Duration::from_secs(30))
                .header("host", "www.kbb.com")
                .header("user-agent", format!("cardata/{}", VERSION));
            let result = builder.send().await;
            match result {
                Err(e) => {
                    last_err = e.into();
                    self.client = reqwest::Client::new();
                    if i + 1 < self.num_retries {
                        sleep(Duration::from_secs(10)).await;
                    }
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
    ) -> Pin<Box<dyn Send + Future<Output = Result<Self::Output, Self::Err>>>>;
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
    ) -> Pin<Box<dyn Send + Future<Output = anyhow::Result<Self::Output>>>> {
        let id = self.0.clone();
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

            let doc_info = extract_doc_json(&doc)?;
            let inventory_item =
                <HashMap<String, Value>>::extract_from_json(&doc_info, "initialState.inventory")
                    .ok()
                    .and_then(|x| x.into_values().next());

            Ok(Some(Listing {
                website: "kbb.com".to_owned(),
                website_id: id,
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
                    .and_then(|x| x.parse().ok())
                },
                engine_description: inventory_item
                    .as_ref()
                    .and_then(|x| String::extract_from_json(&x, "engine").ok()),
                exterior_color: {
                    inventory_item
                        .as_ref()
                        .and_then(|x| String::extract_from_json(&x, "exteriorColorSimple").ok())
                        .or_else(|| {
                            <Vec<String>>::extract_from_json(
                                &doc_info,
                                "initialState.birf.pageData.page.vehicle.color",
                            )
                            .ok()
                            .and_then(vec_into_first)
                        })
                },
                interior_color: {
                    inventory_item
                        .as_ref()
                        .and_then(|x| String::extract_from_json(&x, "interiorColorSimple").ok())
                },
                drive_type: {
                    inventory_item
                        .as_ref()
                        .and_then(|x| String::extract_from_json(&x, "driveGroup").ok())
                        .and_then(|x| x.parse().ok())
                },
                fuel_type: {
                    inventory_item
                        .as_ref()
                        .and_then(|x| String::extract_from_json(&x, "fuelType").ok())
                        .and_then(|x| x.parse().ok())
                },
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
                comments: {
                    inventory_item
                        .as_ref()
                        .and_then(|x| {
                            String::extract_from_json(&x, "additionalInfo.vehicleDescription").ok()
                        })
                        .map(|x| x.replace("<br>", "\n"))
                        .map(|x| inner_text(&Html::parse_fragment(&x).root_element()))
                },
                image_urls: {
                    inventory_item
                        .as_ref()
                        .and_then(|x| extract_image_urls(x).ok())
                },
            }))
        })
    }
}

fn extract_doc_json(body: &Html) -> anyhow::Result<serde_json::Value> {
    let preamble = "window.__BONNET_DATA__=";
    for x in body.select(&Selector::parse("script").unwrap()) {
        let contents = inner_text(&x);
        if !contents.starts_with(preamble) {
            continue;
        }
        return Ok(serde_json::from_str(&contents[preamble.len()..])?);
    }
    Err(anyhow::Error::msg("could not find JSON data in document"))
}

fn extract_image_urls(inventory_item: &Value) -> anyhow::Result<Vec<String>> {
    let mut raw_result = <Vec<Value>>::extract_from_json(inventory_item, "images.sources")?;

    // Re-order so that the primary image URL is first.
    if let Ok(primary) = u64::extract_from_json(inventory_item, "images.primary") {
        let primary = primary as usize;
        if primary < raw_result.len() {
            let x = raw_result.remove(primary);
            raw_result.insert(0, x);
        }
    }

    Ok(raw_result
        .into_iter()
        .filter_map(|x| String::extract_from_json(&x, "src").ok())
        .map(|x| {
            if x.starts_with("//") {
                format!("https://{}", x)
            } else {
                x
            }
        })
        .collect())
}

fn vec_into_first<T>(list: Vec<T>) -> Option<T> {
    for x in list {
        return Some(x);
    }
    None
}
