// Adapted from map-dump:
// https://github.com/unixpickle/map-dump/blob/e5997309cd40a32c63d5fa461746d9dabc1dfea2/src/task_queue.rs

use std::{mem::take, ops::DerefMut, sync::Arc};

use tokio::sync::Mutex;

pub struct TaskQueue<T: Send> {
    queue: Arc<Mutex<Vec<T>>>,
    orig_len: usize,
}

impl<T: Send> Clone for TaskQueue<T> {
    fn clone(&self) -> TaskQueue<T> {
        TaskQueue {
            queue: self.queue.clone(),
            orig_len: self.orig_len,
        }
    }
}

impl<T: Send, I: IntoIterator<Item = T>> From<I> for TaskQueue<T> {
    fn from(x: I) -> TaskQueue<T> {
        let v: Vec<_> = x.into_iter().collect();
        let orig_len = v.len();
        let queue = Arc::new(Mutex::new(v));
        TaskQueue {
            queue: queue,
            orig_len: orig_len,
        }
    }
}

impl<T: Send> TaskQueue<T> {
    pub async fn pop(&self) -> Option<(T, usize)> {
        let mut locked = self.queue.lock().await;
        locked.pop().map(|x| (x, locked.len()))
    }

    pub fn orig_len(&self) -> usize {
        self.orig_len
    }

    pub async fn filter<P: FnMut(&T) -> bool>(&self, p: P) {
        let mut locked = self.queue.lock().await;
        *locked = take(locked.deref_mut()).into_iter().filter(p).collect();
    }
}
