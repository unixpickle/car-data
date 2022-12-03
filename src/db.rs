use crate::chan_util::recv_at_least_one_blocking;
use std::fmt::Write;
use std::path::Path;

use async_channel::{bounded, Receiver, Sender};
use rusqlite::{Connection, Transaction};
use sha2::Digest;
use tokio::task::spawn_blocking;

use crate::types::{Listing, OwnerInfo};

#[derive(Clone)]
pub struct Database {
    req_chan: Sender<
        Box<
            dyn Send
                + FnOnce(
                    anyhow::Result<&mut Transaction>,
                ) -> Box<dyn Send + FnOnce(anyhow::Result<()>)>,
        >,
    >,
}

impl Database {
    pub async fn open<P: AsRef<Path>>(path: P) -> anyhow::Result<Database> {
        let path = path.as_ref().to_owned();
        spawn_blocking(move || -> anyhow::Result<Database> {
            let conn = Connection::open(path)?;
            Database::new_with_conn(conn)
        })
        .await?
    }

    #[allow(dead_code)]
    pub async fn open_in_memory() -> anyhow::Result<Database> {
        spawn_blocking(move || -> anyhow::Result<Database> {
            let conn = Connection::open_in_memory()?;
            Database::new_with_conn(conn)
        })
        .await?
    }

    fn new_with_conn(conn: Connection) -> anyhow::Result<Database> {
        create_tables(&conn)?;
        let (tx, rx) = bounded(100);
        spawn_blocking(move || Database::transaction_worker(conn, rx));
        Ok(Database { req_chan: tx })
    }

    pub async fn check_attempt(
        &self,
        website: &str,
        website_id: &str,
    ) -> anyhow::Result<Option<bool>> {
        let website = website.to_owned();
        let website_id = website_id.to_owned();
        self.with_conn(move |conn| {
            let mut stmt =
                conn.prepare("SELECT success FROM attempt_ids WHERE website=?1 AND website_id=?2")?;
            let mut result_it = stmt.query_map::<bool, _, _>((&website, &website_id), |x| {
                Ok(x.get::<_, i8>(0)? == 1)
            })?;
            Ok(match result_it.next() {
                None => None,
                Some(x) => Some(x?),
            })
        })
        .await
    }

    pub async fn add_failed_attempt(&self, website: &str, website_id: &str) -> anyhow::Result<()> {
        let website = website.to_owned();
        let website_id = website_id.to_owned();
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT OR IGNORE INTO attempt_ids (website, website_id, success) VALUES (?1, ?2, ?3)",
                (&website, &website_id, 0),
            )?;
            Ok(())
        })
        .await
    }

    pub async fn add_listing(&self, listing: Listing) -> anyhow::Result<Option<i64>> {
        self.with_conn(move |conn| {
            let tx = conn.savepoint()?;
            if tx.execute("INSERT OR IGNORE INTO attempt_ids (website, website_id, success) VALUES (?1, ?2, 1)", (&listing.website, &listing.website_id))? != 1 {
                return Ok(None);
            }
            tx.execute(
                "INSERT INTO listings (website, website_id, title, price, make, model, year, odometer, engine, exterior_color, interior_color, drive_type, fuel_type, fuel_economy_0, fuel_economy_1, vin, stock_number, comments) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)",
                rusqlite::params![
                    &listing.website,
                    &listing.website_id,
                    &listing.title,
                    &listing.price,
                    &listing.make,
                    &listing.model,
                    &listing.year,
                    &listing.odometer,
                    &listing.engine_description,
                    &listing.exterior_color,
                    &listing.interior_color,
                    &listing.drive_type,
                    &listing.fuel_type,
                    &maybe_list_entry(&listing.fuel_economy, 0),
                    &maybe_list_entry(&listing.fuel_economy, 1),
                    &listing.vin,
                    &listing.stock_number,
                    &listing.comments
                ],
            )?;
            let last_id = tx.last_insert_rowid();
            if let Some(image_urls) = &listing.image_urls {
                for (i, image_url) in image_urls.iter().enumerate() {
                    tx.execute(
                        "INSERT INTO images (listing_id, image_index, url, hash) VALUES (?1, ?2, ?3, ?4)",
                        rusqlite::params![&last_id, &i, &image_url, &hash_image_url(&image_url)],
                    )?;
                }
            }
            if let Some(owners) = &listing.owners {
                for (i, owner) in owners.iter().enumerate() {
                    tx.execute(
                        "INSERT INTO owners (listing_id, owner_index, website_id, name, website) VALUES (?1, ?2, ?3, ?4, ?5)",
                        rusqlite::params![&last_id, &i, &owner.id, &owner.name, &owner.website],
                    )?;
                }
            }
            tx.commit()?;
            Ok(Some(last_id))
        }).await
    }

    #[allow(dead_code)]
    pub async fn listing_for_id(&self, id: i64) -> anyhow::Result<Option<Listing>> {
        self.with_conn(move |tx| Ok(retrieve_listing(tx, id)?))
            .await
    }

    pub async fn insert_phashes(
        &self,
        hash_and_phash: Vec<(String, String)>,
    ) -> anyhow::Result<()> {
        self.with_conn(move |conn| {
            let tx = conn.savepoint()?;
            for (image_hash, phash) in hash_and_phash {
                tx.execute(
                    "INSERT OR IGNORE INTO phashes (hash, phash, hash_count) VALUES
                     (?1, ?2, (SELECT COUNT(*) from images WHERE hash=?1))",
                    (&image_hash, &phash),
                )?;
            }
            tx.commit()?;
            Ok(())
        })
        .await
    }

    pub async fn counts(&self) -> anyhow::Result<(i64, i64)> {
        self.with_conn(move |tx| {
            let listing_count: i64 =
                tx.query_row("SELECT COUNT(*) FROM listings", (), |row| row.get(0))?;
            let attempt_count: i64 =
                tx.query_row("SELECT COUNT(*) FROM attempt_ids", (), |row| row.get(0))?;
            Ok((listing_count, attempt_count))
        })
        .await
    }

    async fn with_conn<
        T: 'static + Send,
        F: 'static + Send + FnOnce(&mut Transaction) -> anyhow::Result<T>,
    >(
        &self,
        f: F,
    ) -> anyhow::Result<T> {
        let (res_tx, res_rx) = bounded(1);
        let res = self
            .req_chan
            .send(Box::new(move |maybe_tx| match maybe_tx {
                Ok(tx) => {
                    let res = f(tx);
                    Box::new(move |commit_res| {
                        if res.is_ok() && !commit_res.is_ok() {
                            res_tx.send_blocking(Err(commit_res.unwrap_err())).ok();
                        } else {
                            res_tx.send_blocking(res).ok();
                        }
                    })
                }
                Err(e) => Box::new(move |_| {
                    res_tx.send_blocking(Err(e)).ok();
                }),
            }))
            .await;
        if res.is_err() {
            // The true error contains the argument we tried to send,
            // which we cannot wrap in anyhow for some reason.
            Err(anyhow::Error::msg("connection worker has died"))
        } else {
            res_rx.recv().await?
        }
    }

    fn transaction_worker(
        mut conn: Connection,
        rx: Receiver<
            Box<
                dyn Send
                    + FnOnce(
                        anyhow::Result<&mut Transaction>,
                    ) -> Box<dyn Send + FnOnce(anyhow::Result<()>)>,
            >,
        >,
    ) {
        while let Some(reqs) = recv_at_least_one_blocking(&rx) {
            match conn.transaction() {
                Ok(mut tx) => {
                    let mut done_fns = Vec::new();
                    for req in reqs {
                        done_fns.push(req(Ok(&mut tx)));
                    }
                    if let Err(e) = tx.commit() {
                        let msg = format!("{}", e);
                        for done_fn in done_fns {
                            done_fn(Err(anyhow::Error::msg(msg.clone())));
                        }
                    } else {
                        for done_fn in done_fns {
                            done_fn(Ok(()));
                        }
                    }
                }
                Err(e) => {
                    let msg = format!("{}", e);
                    for req in reqs {
                        req(Err(anyhow::Error::msg(msg.clone())))(Ok(()))
                    }
                }
            }
        }
    }
}

pub fn hash_image_url(url: &str) -> String {
    let mut hasher = sha2::Sha256::new();
    hasher.update(url);
    let mut res = String::with_capacity(64);
    for ch in hasher.finalize() {
        write!(&mut res, "{:02x}", ch).unwrap();
    }
    res
}

fn create_tables(conn: &Connection) -> anyhow::Result<()> {
    conn.execute(
        "CREATE TABLE if not exists attempt_ids (
            website     CHAR(16),
            website_id  CHAR(64),
            success     INT,
            PRIMARY KEY (website, website_id)
        )",
        (),
    )?;
    conn.execute(
        "CREATE TABLE if not exists listings (
            id             INTEGER PRIMARY KEY,
            website        TEXT not null,
            website_id     TEXT not null,
            title          TEXT,
            price          TEXT,
            make           TEXT,
            model          TEXT,
            year           INT,
            odometer       TEXT,
            engine         TEXT,
            exterior_color TEXT,
            interior_color TEXT,
            drive_type     TEXT,
            fuel_type      TEXT,
            fuel_economy_0 TEXT,
            fuel_economy_1 TEXT,
            vin            TEXT,
            stock_number   TEXT,
            comments       TEXT
        )",
        (),
    )?;
    conn.execute(
        "CREATE TABLE if not exists owners (
            id             INTEGER PRIMARY KEY,
            listing_id     INT not null,
            owner_index    INT not null,
            website_id     TEXT,
            name           TEXT,
            website        TEXT
        )",
        (),
    )?;
    conn.execute(
        "CREATE TABLE if not exists images (
            id             INTEGER PRIMARY KEY,
            listing_id     INT not null,
            image_index    INT not null,
            url            TEXT not null,
            hash           CHAR(64) not null
        )",
        (),
    )?;
    conn.execute(
        "CREATE TABLE if not exists phashes (
            id             INTEGER PRIMARY KEY,
            hash           CHAR(64) not null,
            hash_count     INT not null,
            phash          CHAR(64) not null,
            UNIQUE (hash)
        )",
        (),
    )?;
    conn.execute(
        "CREATE INDEX if not exists phashindex ON phashes(phash)",
        (),
    )?;
    conn.execute(
        "CREATE INDEX if not exists phasheshashindex ON phashes(hash)",
        (),
    )?;
    conn.execute(
        "CREATE INDEX if not exists imageshashindex ON images(hash)",
        (),
    )?;
    Ok(())
}

fn maybe_list_entry<T>(x: &Option<Vec<T>>, i: usize) -> Option<&T> {
    x.as_ref().and_then(|v| v.get(i))
}

fn maybe_build_list(x: Option<String>, y: Option<String>) -> Option<Vec<String>> {
    if let Some(x) = x {
        let mut res = vec![x];
        if let Some(y) = y {
            res.push(y);
        }
        Some(res)
    } else {
        None
    }
}

fn retrieve_listing(tx: &Transaction, id: i64) -> rusqlite::Result<Option<Listing>> {
    let row = tx.query_row_and_then(
        "SELECT website, website_id, title, price, make, model, year, odometer, engine, exterior_color, interior_color, drive_type, fuel_type, fuel_economy_0, fuel_economy_1, vin, stock_number, comments FROM listings WHERE id=?1",
        (id,),
        |row| -> rusqlite::Result<Listing> {
            Ok(Listing{
                website: row.get(0)?,
                website_id: row.get(1)?,
                title: row.get(2)?,
                price: row.get(3)?,
                make: row.get(4)?,
                model: row.get(5)?,
                year: row.get(6)?,
                odometer: row.get(7)?,
                engine_description: row.get(8)?,
                exterior_color: row.get(9)?,
                interior_color: row.get(10)?,
                drive_type: row.get(11)?,
                fuel_type: row.get(12)?,
                fuel_economy: maybe_build_list(row.get(13)?, row.get(14)?),
                owners: None,
                vin: row.get(15)?,
                stock_number: row.get(16)?,
                comments: row.get(17)?,
                image_urls: None,
            })
        },
    );
    match row {
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
        Ok(mut x) => {
            let mut images = Vec::new();
            for row in tx
                .prepare("SELECT url FROM images WHERE listing_id=?1 ORDER BY image_index")?
                .query_map((&id,), |x| Ok(x.get::<_, String>(0)?))?
            {
                images.push(row?);
            }
            if images.len() > 0 {
                x.image_urls = Some(images);
            }
            let mut owners = Vec::new();
            for row in tx
                .prepare("SELECT website_id, name, website FROM owners WHERE listing_id=?1 ORDER BY owner_index")?
                .query_map((&id,), |x| Ok(OwnerInfo{id: x.get(0)?, name: x.get(1)?, website: x.get(2)?}))?
            {
                owners.push(row?);
            }
            if owners.len() > 0 {
                x.owners = Some(owners);
            }
            Ok(Some(x))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{Listing, OwnerInfo};

    use super::Database;

    #[test]
    fn attempt_ids() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let db = Database::open_in_memory().await.unwrap();
            assert_eq!(db.check_attempt("kbb", "123").await.unwrap(), None);
            assert_eq!(db.check_attempt("kbb", "321").await.unwrap(), None);
            db.add_failed_attempt("kbb", "123").await.unwrap();
            assert_eq!(db.check_attempt("kbb", "123").await.unwrap(), Some(false));
            assert_eq!(db.check_attempt("kbb", "321").await.unwrap(), None);
            assert_eq!(db.check_attempt("kbb_v2", "123").await.unwrap(), None);
            db.add_failed_attempt("kbb_v2", "321").await.unwrap();
            assert_eq!(db.check_attempt("kbb", "123").await.unwrap(), Some(false));
            assert_eq!(db.check_attempt("kbb", "321").await.unwrap(), None);
            assert_eq!(
                db.check_attempt("kbb_v2", "321").await.unwrap(),
                Some(false)
            );
        });
    }

    #[test]
    fn add_listing() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let listing = Listing {
                website: "kbb".to_owned(),
                website_id: "321".to_owned(),
                title: "Car Listing".to_owned(),
                price: Some("$12.98".parse().unwrap()),
                make: Some("Nissan".to_owned()),
                model: Some("Altima".to_owned()),
                year: Some(2019),
                odometer: Some("52 mi".parse().unwrap()),
                engine_description: Some("fast boi".to_owned()),
                exterior_color: Some("Red".to_owned()),
                interior_color: None,
                drive_type: Some("RWD".parse().unwrap()),
                fuel_type: Some("Gasoline".parse().unwrap()),
                fuel_economy: Some(vec!["hello".to_owned(), "world".to_owned()]),
                owners: Some(vec![OwnerInfo {
                    id: "1".to_owned(),
                    name: Some("Annabelle".to_owned()),
                    website: Some("corgi.com/foo".to_owned()),
                }]),
                vin: Some("123123123".to_owned()),
                stock_number: Some("123".to_owned()),
                comments: Some("this car is awesome".to_owned()),
                image_urls: Some(vec!["hello.com".to_owned(), "baz.com".to_owned()]),
            };

            let db = Database::open_in_memory().await.unwrap();
            assert_eq!(db.check_attempt("kbb", "123").await.unwrap(), None);
            assert_eq!(db.check_attempt("kbb", "321").await.unwrap(), None);
            db.add_failed_attempt("kbb", "123").await.unwrap();
            let listing_id = db.add_listing(listing.clone()).await.unwrap().unwrap();
            assert_eq!(db.check_attempt("kbb", "123").await.unwrap(), Some(false));
            assert_eq!(db.check_attempt("kbb", "321").await.unwrap(), Some(true));
            assert_eq!(db.listing_for_id(listing_id + 1).await.unwrap(), None);
            assert_eq!(db.listing_for_id(listing_id).await.unwrap(), Some(listing));
        });
    }
}
