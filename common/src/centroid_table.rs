use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// In-memory routing table mapping centroid ID → shard network address.
///
/// Built once after k-means clustering and then treated as immutable. Share
/// across threads by wrapping in [`std::sync::Arc`]:
///
/// ```rust
/// use std::sync::Arc;
/// use common::CentroidTable;
///
/// let table = Arc::new(CentroidTable::new([(0, "127.0.0.1:7001".into())]));
/// let t2 = Arc::clone(&table);
/// ```
///
/// # Concurrency
///
/// `CentroidTable` is `Send + Sync`. After construction it is read-only, so
/// `Arc<CentroidTable>` is safe for concurrent reads with no locking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentroidTable {
    entries: HashMap<u32, String>,
}

// Compile-time proof that CentroidTable satisfies Send + Sync.
const fn _assert_send_sync<T: Send + Sync>() {}
const _: () = _assert_send_sync::<CentroidTable>();

impl CentroidTable {
    /// Constructs a `CentroidTable` from an iterator of `(centroid_id, shard_address)` pairs.
    ///
    /// If the iterator yields duplicate centroid IDs, the last value wins. An
    /// empty iterator produces a valid empty table.
    ///
    /// # Panics
    ///
    /// Never panics.
    pub fn new(entries: impl IntoIterator<Item = (u32, String)>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
        }
    }

    /// Returns the shard address for the given centroid ID, or `None` if not
    /// registered.
    ///
    /// Lookup is O(1) amortized.
    ///
    /// # Panics
    ///
    /// Never panics.
    pub fn get(&self, centroid_id: u32) -> Option<&str> {
        self.entries.get(&centroid_id).map(String::as_str)
    }

    /// Returns the number of centroid-to-shard mappings in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the table contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serializes the table to a JSON string.
    ///
    /// The JSON shape is:
    /// ```json
    /// { "entries": { "<centroid_id>": "<host:port>", ... } }
    /// ```
    /// Centroid IDs are rendered as decimal string keys. Key order is
    /// non-deterministic (hash-map iteration order).
    ///
    /// # Errors
    ///
    /// Returns [`serde_json::Error`] if serialization fails. In practice this
    /// is infallible for `CentroidTable`.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserializes a `CentroidTable` from a JSON string produced by
    /// [`CentroidTable::to_json`].
    ///
    /// # Errors
    ///
    /// Returns [`serde_json::Error`] if `json` is not valid JSON or does not
    /// match the expected shape.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serializes the table to a compact binary (bincode) encoding.
    ///
    /// # Errors
    ///
    /// Returns [`bincode::Error`] if serialization fails. In practice this is
    /// infallible for `CentroidTable`.
    pub fn to_bincode(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserializes a `CentroidTable` from a bincode byte slice produced by
    /// [`CentroidTable::to_bincode`].
    ///
    /// # Errors
    ///
    /// Returns [`bincode::Error`] if `bytes` are malformed, truncated, or do
    /// not represent a valid `CentroidTable`.
    pub fn from_bincode(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}
