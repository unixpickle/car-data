use std::{
    fs::rename,
    path::{Path, PathBuf},
};

use crate::{
    chan_util::recv_at_least_one,
    db::Database,
    image_util::{downsample_image, hash_image},
};
use clap::Parser;
use image::ImageFormat;
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
    println!("creating output directories...");

    let chars = "0123456789abcdef";
    for x in chars.chars() {
        for y in chars.chars() {
            let full_path: PathBuf = [&args.output_dir, &format!("{}{}", x, y)].iter().collect();
            create_dir_all(full_path).await?;
        }
    }

    println!("running dedup...");

    let (path_tx, path_rx) = async_channel::bounded(args.concurrency);
    let image_dir = args.image_dir.clone();
    spawn(async move {
        for x in chars.chars() {
            for y in chars.chars() {
                let sub_dir: PathBuf = [&image_dir, &format!("{}{}", x, y)].iter().collect();
                let mut reader = read_dir(sub_dir).await.unwrap();
                while let Some(path_info) = reader.next_entry().await.unwrap() {
                    path_tx.send(path_info.path()).await.unwrap();
                }
            }
        }
    });

    let (hash_tx, hash_rx) = async_channel::bounded(100);
    for _ in 0..args.concurrency {
        let path_rx = path_rx.clone();
        let hash_tx = hash_tx.clone();
        let args = args.clone();
        spawn_blocking(move || {
            while let Ok(path) = path_rx.recv_blocking() {
                match hash_and_downsample(&args, &path) {
                    Ok(hash) => hash_tx.send_blocking((path, hash)).unwrap(),
                    Err(e) => eprintln!("error from {:?}: {}", path, e),
                }
            }
        });
    }
    drop(hash_tx);

    let db = Database::open(&args.db_path).await?;
    let mut num_inserted: u64 = 0;
    while let Some(objs) = recv_at_least_one(&hash_rx).await {
        let batch_size = objs.len();
        db.insert_phashes(
            objs.into_iter()
                .map(|(path, phash)| (path_basename(&path), phash))
                .collect(),
        )
        .await?;
        for _ in 0..batch_size {
            num_inserted += 1;
            if num_inserted % 100 == 0 {
                println!("inserted {} hashes", num_inserted);
            }
        }
    }
    Ok(())
}

fn hash_and_downsample(args: &Args, path: &Path) -> anyhow::Result<String> {
    let mut img = image::io::Reader::open(path)?
        .with_guessed_format()?
        .decode()?;
    let hash = hash_image(args.hash_resolution, &img);
    img = downsample_image(args.out_resolution, img);

    let out_path: PathBuf = [&args.output_dir, &hash[0..2], &hash].iter().collect();
    let tmp_out_path: PathBuf = [
        &args.output_dir,
        &hash[0..2],
        &format!("tmp_{}", path_basename(path)),
    ]
    .iter()
    .collect();
    img.save_with_format(&tmp_out_path, ImageFormat::Jpeg)?;
    rename(tmp_out_path, out_path)?;
    Ok(hash)
}

fn path_basename(p: &Path) -> String {
    p.file_name().unwrap().to_string_lossy().into_owned()
}
