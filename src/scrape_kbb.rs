use clap::Parser;

#[derive(Clone, Parser)]
pub struct Args {
    #[clap(long, value_parser, default_value_t = 660000000)]
    min_id: i64,

    #[clap(long, value_parser, default_value_t = 664000000)]
    max_id: i64,

    #[clap(short, long, value_parser, default_value_t = 15)]
    num_retries: i32,

    #[clap(short, long, value_parser, default_value_t = 8)]
    concurrency: usize,

    #[clap(value_parser)]
    db_path: String,

    #[clap(value_parser)]
    image_dir: String,
}

pub async fn main(args: Args) -> anyhow::Result<()> {
    Ok(())
}
