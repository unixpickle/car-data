use std::{
    mem::take,
    path::Path,
    sync::{Arc, Mutex},
};

use rusqlite::Connection;
use sha2::Digest;
use tokio::task::spawn_blocking;

use crate::types::Listing;

pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub async fn open<P: AsRef<Path>>(path: P) -> anyhow::Result<Database> {
        let path = path.as_ref().to_owned();
        spawn_blocking(move || -> anyhow::Result<Database> {
            let conn = Connection::open(path)?;
            create_tables(&conn)?;
            Ok(Database {
                conn: Arc::new(Mutex::new(conn)),
            })
        })
        .await?
    }

    pub async fn open_in_memory() -> anyhow::Result<Database> {
        spawn_blocking(move || -> anyhow::Result<Database> {
            let conn = Connection::open_in_memory()?;
            create_tables(&conn)?;
            Ok(Database {
                conn: Arc::new(Mutex::new(conn)),
            })
        })
        .await?
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
            let tx = conn.transaction()?;
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
                        "INSERT INTO owners (listing_id, owner_index, website_id, name, website) VALUES (?1, ?2, ?3, ?4)",
                        rusqlite::params![&last_id, &i, &owner.id, &owner.name, &owner.website],
                    )?;
                }
            }
            tx.commit()?;
            Ok(Some(last_id))
        }).await
    }

    async fn with_conn<
        T: 'static + Send,
        F: 'static + Send + FnOnce(&mut Connection) -> anyhow::Result<T>,
    >(
        &self,
        f: F,
    ) -> anyhow::Result<T> {
        let conn = self.conn.clone();
        spawn_blocking(move || f(&mut conn.lock().unwrap())).await?
    }
}

pub fn hash_image_url(url: &str) -> String {
    let mut hasher = sha2::Sha256::new();
    hasher.update(url);
    format!("{:x?}", hasher.finalize())
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
            id             INT PRIMARY KEY,
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
            id             INT PRIMARY KEY,
            listing_id     INT not null,
            owner_index    INT not null
            website_id     TEXT,
            name           TEXT,
            website        TEXT
        )",
        (),
    )?;
    conn.execute(
        "CREATE TABLE if not exists images (
            id             INT PRIMARY KEY,
            listing_id     INT not null,
            image_index    INT not null,
            url            TEXT not null,
            hash           CHAR(64) not null
        )",
        (),
    )?;
    Ok(())
}

fn maybe_list_entry<T>(x: &Option<Vec<T>>, i: usize) -> Option<&T> {
    x.as_ref().and_then(|v| v.get(i))
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
            db.add_listing(listing.clone()).await.unwrap();
            assert_eq!(db.check_attempt("kbb", "123").await.unwrap(), Some(false));
            assert_eq!(db.check_attempt("kbb", "321").await.unwrap(), Some(true));
        });
    }
}
