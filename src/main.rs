use std::process::ExitCode;

use clap::Parser;

mod chan_util;
mod db;
mod dedup_images;
mod export_data;
mod image_util;
mod kbb;
mod parse_util;
mod scrape_kbb;
mod task_queue;
mod types;

#[derive(Parser, Clone)]
#[clap(author, version, about, long_about = None)]
enum Args {
    ScrapeKbb {
        #[clap(flatten)]
        args: scrape_kbb::Args,
    },
    DedupImages {
        #[clap(flatten)]
        args: dedup_images::Args,
    },
    ExportData {
        #[clap(flatten)]
        args: export_data::Args,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    if let Err(e) = match args {
        Args::ScrapeKbb { args } => scrape_kbb::main(args).await,
        Args::DedupImages { args } => dedup_images::main(args).await,
        Args::ExportData { args } => export_data::main(args).await,
    } {
        eprintln!("{}", e);
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
