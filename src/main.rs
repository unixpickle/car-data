use std::process::ExitCode;

use crate::kbb::{Client, ListingRequest};

mod kbb;
mod parse_util;

#[tokio::main]
async fn main() -> ExitCode {
    let mut client = Client::new(15);
    let listing = client
        // .run(ListingRequest("622683064".to_owned()))
        .run(ListingRequest("662683064".to_owned()))
        .await
        .unwrap();
    println!("listing: {:?}", listing);
    ExitCode::SUCCESS
}
