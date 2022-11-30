use async_channel::Receiver;

pub async fn recv_at_least_one<T>(rx: &Receiver<T>) -> Option<Vec<T>> {
    if let Ok(obj) = rx.recv().await {
        let mut buffer = vec![obj];
        loop {
            match rx.try_recv() {
                Ok(obj) => buffer.push(obj),
                _ => return Some(buffer),
            }
        }
    } else {
        None
    }
}

pub fn recv_at_least_one_blocking<T>(rx: &Receiver<T>) -> Option<Vec<T>> {
    if let Ok(obj) = rx.recv_blocking() {
        let mut buffer = vec![obj];
        loop {
            match rx.try_recv() {
                Ok(obj) => buffer.push(obj),
                _ => return Some(buffer),
            }
        }
    } else {
        None
    }
}
