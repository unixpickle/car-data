use std::{
    mem::take,
    path::Path,
    sync::{Arc, Mutex},
};

use rusqlite::Connection;
use tokio::task::spawn_blocking;

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

    async fn with_conn<
        T: 'static + Send,
        F: 'static + Send + FnOnce(&Connection) -> anyhow::Result<T>,
    >(
        &self,
        f: F,
    ) -> anyhow::Result<T> {
        let conn = self.conn.clone();
        spawn_blocking(move || f(&conn.lock().unwrap())).await?
    }
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
            hash           CHAR(32) not null
        )",
        (),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
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
}
