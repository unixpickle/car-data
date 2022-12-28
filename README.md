# car-data

Let's try to train a model to predict the price of a car given a single photo of that car.

# Current status

I have scraped several hundred thousand listings from [Kelley Blue Book](https://www.kbb.com/), including associated photos for each listing. This ended up being about a terabyte of data, but after compressing and deduplicating the storage requirements are much lower.

I have tried fine-tuning some models on a discretized price prediction task, with limited success.

# Usage

Compile with `cargo build --release`.

## Scraping kbb.com

To scrape `kbb.com` data, run:

```
./target/release/car-data scrape-kbb /path/to/db.db /path/to/images
```

To dedup the downloaded images, run:

```
./target/release/car-data dedup-images \
    /path/to/db.db \
    /path/to/images \
    /path/to/dedup
```
