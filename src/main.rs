use std::process::ExitCode;

use clap::Parser;

mod db;
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
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    if let Err(e) = match args {
        Args::ScrapeKbb { args } => scrape_kbb::main(args).await,
    } {
        eprintln!("{}", e);
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
