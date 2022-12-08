use clap::Parser;
use tokio::task::spawn_blocking;

use npy_writer::NpzWriter;

use crate::{
    db::Database,
    types::{Price, PriceUnit},
};

#[derive(Clone, Parser)]
pub struct Args {
    #[clap(value_parser)]
    db_path: String,

    #[clap(value_parser)]
    output_path: String,
}

pub async fn main(args: Args) -> anyhow::Result<()> {
    let db = Database::open(&args.db_path).await?;
    let results = db.unique_phashes();

    let mut phashes = Vec::new();
    let mut prices = Vec::new();
    let mut makes = Vec::new();
    let mut models = Vec::new();
    let mut years = Vec::new();

    let mut seen = 0usize;
    let mut used = 0usize;

    while let Ok(item) = results.recv().await {
        let (phash, listing) = item?;
        seen += 1;
        if let Some(dollars) = get_dollar_amount(listing.price) {
            used += 1;
            phashes.push(phash);
            prices.push(dollars);
            makes.push(listing.make.unwrap_or_default());
            models.push(listing.model.unwrap_or_default());
            years.push(listing.year.unwrap_or_default());
        }
        if seen % 1000 == 0 {
            print_stats(seen, used);
        }
    }
    print_stats(seen, used);
    spawn_blocking(|| -> anyhow::Result<()> {
        let mut writer = NpzWriter::new(args.output_path)?;
        writer.write("phashes", phashes)?;
        writer.write("prices", prices)?;
        writer.write("makes", makes)?;
        writer.write("models", models)?;
        writer.write("years", years)?;
        writer.close()?;
        Ok(())
    })
    .await??;
    Ok(())
}

fn print_stats(seen: usize, used: usize) {
    println!(
        "total={} used={} (frac={:.02}%)",
        seen,
        used,
        100.0 * (used as f64) / (seen as f64),
    );
}

fn get_dollar_amount(price: Option<Price>) -> Option<f64> {
    if let Some(p) = price {
        if p.unit == PriceUnit::Cents && p.value > 0 {
            return Some((p.value as f64) / 100.0);
        }
    }
    None
}
