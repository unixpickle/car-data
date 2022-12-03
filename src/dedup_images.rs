use sha2::Digest;
use std::{
    fmt::Write,
    fs::rename,
    path::{Path, PathBuf},
};

use crate::{chan_util::recv_at_least_one, db::Database};
use clap::Parser;
use image::{imageops::FilterType, DynamicImage, EncodableLayout, ImageFormat};
use tokio::{
    fs::{create_dir_all, read_dir},
    spawn,
    task::spawn_blocking,
};

#[derive(Clone, Parser)]
pub struct Args {
    #[clap(long, value_parser, default_value_t = 16)]
    hash_resolution: u32,

    #[clap(long, value_parser, default_value_t = 256)]
    out_resolution: u32,

    #[clap(long, value_parser, default_value_t = 4)]
    concurrency: usize,

    #[clap(value_parser)]
    db_path: String,

    #[clap(value_parser)]
    image_dir: String,

    #[clap(value_parser)]
    output_dir: String,
}

pub async fn main(args: Args) -> anyhow::Result<()> {
    Ok(())
}
