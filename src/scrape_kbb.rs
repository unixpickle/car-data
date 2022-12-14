use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use crate::{
    db::{hash_image_url, Database},
    dedup_images::create_hash_prefixes,
    image_util::downsample_image,
    kbb::{Client, ImageDownloadRequest, ListingRequest},
    task_queue::TaskQueue,
    types::Listing,
};
use clap::Parser;
use image::ImageFormat;
use rand::seq::SliceRandom;
use tokio::{spawn, sync::mpsc::channel, task::spawn_blocking, time::Instant};

const KBB_WEBSITE_NAME: &str = "kbb.com";

#[derive(Clone, Parser)]
pub struct Args {
    #[clap(long, value_parser, default_value_t = 660000000)]
    min_id: i64,

    #[clap(long, value_parser, default_value_t = 668000000)]
    max_id: i64,

    #[clap(short, long, value_parser, default_value_t = 15)]
    num_retries: i32,

    #[clap(short, long, value_parser, default_value_t = 8)]
    concurrency: usize,

    #[clap(short, long, value_parser, default_value_t = 256)]
    resize_images: u32,

    #[clap(value_parser)]
    db_path: String,

    #[clap(value_parser)]
    image_dir: String,
}

pub async fn main(args: Args) -> anyhow::Result<()> {
    create_hash_prefixes(&args.image_dir).await?;

    println!("connecting database...");
    let db = Database::open(&args.db_path).await?;
    println!("creating permutation...");
    let perm = generate_permutation(args.min_id, args.max_id);
    println!("filtering permutation...");
    let used_ids: HashSet<_> = db.get_attempt_ids(KBB_WEBSITE_NAME).await?;
    perm.filter(|x| !used_ids.contains(&format!("{}", x))).await;
    println!("scraping...");

    let (tx, mut rx) = channel(args.concurrency);
    for _ in 0..args.concurrency {
        let local_db = db.clone();
        let local_perm = perm.clone();
        let local_args = args.clone();
        let local_tx = tx.clone();
        spawn(async move {
            local_tx
                .send(fetch_listings(local_db, local_perm, local_args).await)
                .await
                .unwrap();
        });
    }

    while let Some(exc) = rx.recv().await {
        exc?;
    }

    Ok(())
}

async fn fetch_listings(db: Database, perm: TaskQueue<i64>, args: Args) -> anyhow::Result<()> {
    let mut client = Client::new(args.num_retries);
    while let Some((id, remaining)) = perm.pop().await {
        let id_str = format!("{}", id);
        if db.check_attempt(KBB_WEBSITE_NAME, &id_str).await?.is_some() {
            continue;
        }
        if let Some(listing) = client.run(ListingRequest(id_str.clone())).await? {
            download_listing_images(&mut client, &args.image_dir, args.resize_images, &listing)
                .await?;
            db.add_listing(listing).await?;
        } else {
            db.add_failed_attempt(KBB_WEBSITE_NAME, &id_str).await?;
        }

        let completed = perm.orig_len() - remaining;
        if completed % 100 == 0 {
            let start = Instant::now();
            let (num_listings, total_attempts) = db.counts().await?;
            let counts_duration = start.elapsed();
            eprintln!(
                "scraped={:.04}% hit_rate={:.02}% hit_total={} db_latency={:.05}",
                100.0 * (completed as f64) / (perm.orig_len() as f64),
                100.0 * (num_listings as f64) / (total_attempts as f64),
                num_listings,
                counts_duration.as_secs_f64(),
            );
        }
    }
    Ok(())
}

async fn download_listing_images(
    client: &mut Client,
    image_path: &str,
    resize_images: u32,
    listing: &Listing,
) -> anyhow::Result<()> {
    if let Some(urls) = &listing.image_urls {
        for url in urls {
            let image_hash = hash_image_url(&url);
            let out_path: PathBuf = [image_path, &image_hash[0..2], &image_hash]
                .iter()
                .collect();
            if tokio::fs::metadata(&out_path).await.is_ok() {
                // Skip for already-downloaded image URL
                continue;
            }
            // Download+rename to atomically write the file.
            let tmp_out_path: PathBuf = [
                image_path,
                &format!("{}.{}", image_hash, listing.website_id),
            ]
            .iter()
            .collect();
            client
                .run(ImageDownloadRequest {
                    url: url.clone(),
                    out_path: tmp_out_path.clone(),
                })
                .await?;
            if resize_images != 0 {
                spawn_blocking(move || resize_or_rename(resize_images, tmp_out_path, out_path))
                    .await??;
            } else {
                tokio::fs::rename(tmp_out_path, out_path).await?;
            }
        }
    }
    Ok(())
}

fn resize_or_rename<T: AsRef<Path>>(size: u32, src: T, dst: T) -> anyhow::Result<()> {
    if attempt_resize(size, &src, &dst).is_err() {
        std::fs::rename(src, dst)?;
    }
    Ok(())
}

fn attempt_resize<T: AsRef<Path>>(size: u32, src: T, dst: T) -> anyhow::Result<()> {
    let img = downsample_image(
        size,
        image::io::Reader::open(&src)?
            .with_guessed_format()?
            .decode()?,
    );
    let tmp_tmp_path = format!("{}_writing", src.as_ref().to_string_lossy());
    img.save_with_format(&tmp_tmp_path, ImageFormat::Jpeg)?;
    std::fs::rename(tmp_tmp_path, dst)?;
    Ok(())
}

fn generate_permutation(min: i64, max: i64) -> TaskQueue<i64> {
    let mut v: Vec<i64> = (min..max).collect();
    v.shuffle(&mut rand::thread_rng());
    v.into()
}
