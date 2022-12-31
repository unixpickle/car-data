# car-data

Let's try to train a model to predict the price of a car given a single photo of that car.

# Current status

I have scraped several hundred thousand listings from [Kelley Blue Book](https://www.kbb.com/), including associated photos for each listing. This ended up being about a terabyte of data, but after compressing and deduplicating the storage requirements are much lower.

I have tried fine-tuning some models on a discretized price prediction task, with limited success.

# Usage

First, you should:

 * Compile the scraper with `cargo build --release`.
 * Install the Python package with `pip install -e .`.

## Scraping data

To run the scraper, run:

```
./target/release/car-data scrape-kbb /path/to/db.db /path/to/images
```

In the above command, `/path/to/db.db` is the path where the metadata will be saved. It is stored as a sqlite3 database. The `/path/to/images` directory will be used to dump raw images.

To deduplicate and downsample the downloaded images, run:

```
./target/release/car-data dedup-images \
    /path/to/db.db \
    /path/to/images \
    /path/to/dedup
```

From here on out, we will use the `/path/to/dedup` directory instead of `/path/to/images`, since the former directory contains all of the images we will actually use for training.

To export the resulting dataset as a `.npz` file to load in Python, run:

```
./target/release/car-data export-data \
    /path/to/db.db \
    /path/to/index.npz
```

## Filtering the dataset

To filter the dataset, you will first want to compute feature vectors for the entire dataset. These will be exported as a directory full of npz files with shards of features. You can do this with the following command:

```
python3 -m car_data.scripts.clip_features \
    /path/to/dedup \
    /path/to/features
```

Once you have labeled some images for the filter, you can train it quickly like so:

```
python3 -m car_data.scripts.train_filter \
    --positive_dirs /path/to/positive_dir \
    --negative_dirs /path/to/negative_dir \
    --model_out /path/to/filter.pt
```

To filter the dataset `.npz` file using the filter, you can use this command:

```
python3 -m car_data.scripts.filter_index \
    --index /path/to/index.npz \
    --feature_dir /path/to/features \
    --classifier_path /path/to/filter.pt \
    --output_path /path/to/index_filtered.npz
```

## Training a model

To train a MobileNetV2 with auxiliary losses:

```
python3 -m car_data.scripts.train \
    --index_path /path/to/index_filtered.npz \
    --image_dir /path/to/dedup \
    --save_dir /path/to/mobilenetv2_save_dir \
    --lr 1e-4 \
    --batch_size 64 \
    --eval_interval 1 \
    --use_data_aug \
    --model mobilenetv2
```

To finetune CLIP with auxiliary losses:

```
python3 -m car_data.scripts.train \
    --index_path /path/to/index_filtered.npz \
    --image_dir /path/to/dedup \
    --save_dir /path/to/clip_save_dir \
    --lr 1e-5 \
    --batch_size 64 \
    --microbatch 16 \
    --eval_interval 1 \
    --use_data_aug \
    --model clip
```
