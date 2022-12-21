use crate::db::Database;
use clap::Parser;

#[derive(Clone, Parser)]
pub struct Args {
    #[clap(value_parser)]
    db_path: String,
}

pub async fn main(args: Args) -> anyhow::Result<()> {
    let db = Database::open(args.db_path).await?;
    let counts = db.make_model_counts().await?;

    let total: i64 = counts.iter().map(|(_, _, count)| count).sum();

    for i in 1..10 {
        let sub_idx = (((counts.len() * i) as f64) / 10.0).round() as usize;
        let sub_total: i64 = counts[0..sub_idx].iter().map(|(_, _, count)| count).sum();
        let (_, _, sub_count) = &counts[sub_idx];
        println!(
            "{}-percentile: {:.02}% (total items: {}) (per-entry {})",
            i * 10,
            (sub_total as f64) / (total as f64) * 100.0,
            sub_idx,
            sub_count
        );
    }

    Ok(())
}
