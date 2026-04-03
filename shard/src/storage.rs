use std::collections::BinaryHeap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::mem;
use std::path::{Path, PathBuf};

use common::{SearchResult, Vector, VectorId};
use memmap2::MmapMut;

use crate::distance::{self, DistanceMetric};

const MAGIC: [u8; 8] = *b"PXVEC001";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 32;
const MAGIC_OFFSET: usize = 0;
const VERSION_OFFSET: usize = 8;
const DIMENSION_OFFSET: usize = 12;
const LEN_OFFSET: usize = 16;
const CAPACITY_OFFSET: usize = 24;
const VECTOR_ID_SIZE: usize = mem::size_of::<u64>();
const FLOAT_SIZE: usize = mem::size_of::<f32>();

/// Errors that can occur when reading from or writing to a [`FlatVectorStore`].
#[derive(Debug)]
pub enum StorageError {
    /// An underlying I/O error.
    Io(io::Error),
    /// The file header is missing, truncated, or contains unexpected values.
    InvalidHeader(&'static str),
    /// The dimension of a vector does not match the store's configured dimension.
    DimensionMismatch { expected: usize, actual: usize },
    /// A record's byte layout is not correctly aligned or sized.
    CorruptRecordLayout,
    /// An arithmetic overflow occurred while computing record or file sizes.
    RecordCountOverflow,
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "storage I/O error: {error}"),
            Self::InvalidHeader(message) => write!(f, "invalid storage header: {message}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "vector dimension mismatch: expected {expected}, got {actual}")
            }
            Self::CorruptRecordLayout => write!(f, "corrupt storage record layout"),
            Self::RecordCountOverflow => write!(f, "record count overflow"),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<io::Error> for StorageError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// A zero-copy view into one record inside a [`FlatVectorStore`].
///
/// The lifetime `'a` is tied to the memory-mapped region of the store, so no
/// heap allocation is needed when reading individual vectors.
#[derive(Debug, Clone, Copy)]
pub struct VectorRecordRef<'a> {
    /// The unique identifier of this vector.
    pub id: VectorId,
    /// The raw float data for this vector, borrowed from the memory map.
    pub data: &'a [f32],
}

/// A flat, memory-mapped vector store backed by a single binary file.
///
/// Vectors are stored sequentially in a fixed-dimension format with a 32-byte
/// header. The file grows automatically (doubling strategy) when capacity is
/// exhausted. Use [`FlatVectorStore::open_or_create`] to obtain an instance;
/// the file is created if it does not yet exist and re-opened with its existing
/// contents otherwise.
///
/// # File layout
///
/// ```text
/// [0..8]   magic bytes  "PXVEC001"
/// [8..12]  version      u32 LE
/// [12..16] dimension    u32 LE
/// [16..24] len          u64 LE  (current number of records)
/// [24..32] capacity     u64 LE  (allocated slots)
/// [32..]   records      (id: u64 LE)(f32 * dimension) * capacity
/// ```
#[derive(Debug)]
pub struct FlatVectorStore {
    path: PathBuf,
    file: File,
    mmap: MmapMut,
    dimension: usize,
    len: usize,
    capacity: usize,
    record_size: usize,
}

impl FlatVectorStore {
    /// Opens an existing store file or creates a new one at `path`.
    ///
    /// If the file does not exist or is empty it is initialised with the given
    /// `dimension` and `initial_capacity`. If the file already exists its
    /// header is validated and its contents are memory-mapped for access.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when:
    /// - `dimension` is zero.
    /// - The file cannot be opened or created.
    /// - An existing file's header is corrupt, has an unsupported version, or
    ///   its recorded dimension does not match the requested `dimension`.
    pub fn open_or_create(
        path: impl AsRef<Path>,
        dimension: usize,
        initial_capacity: usize,
    ) -> Result<Self, StorageError> {
        if dimension == 0 {
            return Err(StorageError::InvalidHeader("dimension must be greater than zero"));
        }

        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let existing_len = file.metadata()?.len();
        let record_size = record_size_for(dimension)?;

        if existing_len == 0 {
            let capacity = initial_capacity.max(1);
            initialize_file(&file, dimension, capacity, record_size)?;
        }

        let mmap = map_file(&file)?;
        let mut store = Self {
            path,
            file,
            mmap,
            dimension,
            len: 0,
            capacity: 0,
            record_size,
        };

        store.reload_metadata()?;
        Ok(store)
    }

    /// Returns the number of vectors currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the store contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the fixed embedding dimension of this store.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of vector slots allocated in the backing file.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the path of the backing file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends a single vector to the store.
    ///
    /// The store grows its backing file automatically when needed.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::DimensionMismatch`] if `vector.data.len()` does
    /// not equal [`Self::dimension`].
    pub fn insert(&mut self, vector: &Vector) -> Result<(), StorageError> {
        self.insert_raw(vector.id, &vector.data)
    }

    /// Appends multiple vectors to the store in a single operation.
    ///
    /// All dimension checks are performed before any writes so that the store
    /// is never partially updated on error. The backing file grows as needed.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::DimensionMismatch`] if any vector's data length
    /// does not equal [`Self::dimension`].
    pub fn insert_batch<'a>(
        &mut self,
        vectors: impl IntoIterator<Item = &'a Vector>,
    ) -> Result<(), StorageError> {
        let vectors: Vec<&Vector> = vectors.into_iter().collect();

        for vector in &vectors {
            self.validate_dimension(vector.data.len())?;
        }

        let additional = vectors.len();
        if additional == 0 {
            return Ok(());
        }

        self.ensure_capacity(self.len + additional)?;
        for vector in vectors {
            self.write_record(self.len, vector.id, &vector.data)?;
            self.len += 1;
        }
        self.write_len(self.len)?;
        Ok(())
    }

    /// Retrieves the vector at `index`, allocating a new [`Vector`] on the heap.
    ///
    /// Returns `Ok(None)` if `index` is out of bounds.
    pub fn get(&self, index: usize) -> Result<Option<Vector>, StorageError> {
        match self.record_ref(index)? {
            Some(record) => Ok(Some(Vector {
                id: record.id,
                data: record.data.to_vec(),
            })),
            None => Ok(None),
        }
    }

    /// Returns a zero-copy view of the vector at `index` without heap allocation.
    ///
    /// Returns `Ok(None)` if `index` is out of bounds.
    pub fn record_ref(&self, index: usize) -> Result<Option<VectorRecordRef<'_>>, StorageError> {
        if index >= self.len {
            return Ok(None);
        }

        let offset = self.record_offset(index)?;
        let id = read_u64(&self.mmap, offset)?;
        let data_start = offset + VECTOR_ID_SIZE;
        let data_end = data_start + (self.dimension * FLOAT_SIZE);
        let data = bytes_as_f32_slice(&self.mmap[data_start..data_end])?;

        Ok(Some(VectorRecordRef { id, data }))
    }

    /// Returns an iterator over all stored vectors as zero-copy [`VectorRecordRef`] views.
    pub fn iter(&self) -> FlatVectorStoreIter<'_> {
        FlatVectorStoreIter {
            store: self,
            index: 0,
        }
    }

    /// Flushes all pending writes to the backing file.
    ///
    /// This synchronises the memory-mapped region with the filesystem. Call
    /// this before the process exits or before reopening the store from a
    /// different handle.
    pub fn flush(&self) -> Result<(), StorageError> {
        self.mmap.flush()?;
        Ok(())
    }

    /// Returns the `k` nearest vectors to `query` by brute-force scan.
    ///
    /// Every stored record is visited and scored with the chosen `metric`.
    /// Results are returned sorted by ascending distance; ties are broken by
    /// ascending [`VectorId`] for determinism.
    ///
    /// Returns an empty `Vec` when `k == 0` or the store is empty.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::DimensionMismatch`] if `query.len()` does not
    /// equal [`Self::dimension`].
    pub fn search_topk(
        &self,
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>, StorageError> {
        self.validate_dimension(query.len())?;
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        // Max-heap: top entry is the *worst* current candidate.
        // When the heap is full we evict the worst if a better candidate appears.
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);

        for record in self.iter() {
            let d = match metric {
                DistanceMetric::L2 => distance::l2_distance(query, record.data),
                DistanceMetric::Cosine => distance::cosine_distance(query, record.data),
            };
            let candidate = Candidate { distance: d, id: record.id };
            if heap.len() < k {
                heap.push(candidate);
            } else if let Some(worst) = heap.peek()
                && candidate < *worst {
                    heap.pop();
                    heap.push(candidate);
                }
        }

        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|c| SearchResult { id: c.id, distance: c.distance })
            .collect();
        results.sort_unstable_by(|a, b| {
            a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id))
        });
        Ok(results)
    }

    fn insert_raw(&mut self, id: VectorId, data: &[f32]) -> Result<(), StorageError> {
        self.validate_dimension(data.len())?;
        self.ensure_capacity(self.len + 1)?;
        self.write_record(self.len, id, data)?;
        self.len += 1;
        self.write_len(self.len)?;
        Ok(())
    }

    fn validate_dimension(&self, actual: usize) -> Result<(), StorageError> {
        if actual != self.dimension {
            return Err(StorageError::DimensionMismatch {
                expected: self.dimension,
                actual,
            });
        }
        Ok(())
    }

    fn ensure_capacity(&mut self, required_len: usize) -> Result<(), StorageError> {
        if required_len <= self.capacity {
            return Ok(());
        }

        let mut new_capacity = self.capacity.max(1);
        while new_capacity < required_len {
            new_capacity = new_capacity.saturating_mul(2);
            if new_capacity < required_len && new_capacity == usize::MAX {
                return Err(StorageError::RecordCountOverflow);
            }
        }

        self.remap(new_capacity)?;
        self.capacity = new_capacity;
        self.write_capacity(self.capacity)?;
        Ok(())
    }

    fn remap(&mut self, new_capacity: usize) -> Result<(), StorageError> {
        self.mmap.flush()?;
        let new_len = file_len_for(new_capacity, self.record_size)?;
        self.file.set_len(new_len)?;
        self.mmap = map_file(&self.file)?;
        Ok(())
    }

    fn reload_metadata(&mut self) -> Result<(), StorageError> {
        validate_header(&self.mmap)?;

        let stored_dimension = read_u32(&self.mmap, DIMENSION_OFFSET)? as usize;
        if stored_dimension != self.dimension {
            return Err(StorageError::DimensionMismatch {
                expected: self.dimension,
                actual: stored_dimension,
            });
        }

        self.len = read_u64(&self.mmap, LEN_OFFSET)? as usize;
        self.capacity = read_u64(&self.mmap, CAPACITY_OFFSET)? as usize;

        if self.len > self.capacity {
            return Err(StorageError::InvalidHeader("record count exceeds capacity"));
        }

        let expected_len = file_len_for(self.capacity, self.record_size)?;
        if self.file.metadata()?.len() != expected_len {
            return Err(StorageError::InvalidHeader("file length does not match capacity"));
        }

        Ok(())
    }

    fn write_record(&mut self, index: usize, id: VectorId, data: &[f32]) -> Result<(), StorageError> {
        let offset = self.record_offset(index)?;
        write_u64(&mut self.mmap, offset, id)?;

        let data_start = offset + VECTOR_ID_SIZE;
        for (position, value) in data.iter().enumerate() {
            let value_offset = data_start + (position * FLOAT_SIZE);
            write_f32(&mut self.mmap, value_offset, *value)?;
        }

        Ok(())
    }

    fn write_len(&mut self, len: usize) -> Result<(), StorageError> {
        write_u64(&mut self.mmap, LEN_OFFSET, len as u64)
    }

    fn write_capacity(&mut self, capacity: usize) -> Result<(), StorageError> {
        write_u64(&mut self.mmap, CAPACITY_OFFSET, capacity as u64)
    }

    fn record_offset(&self, index: usize) -> Result<usize, StorageError> {
        let record_start = index
            .checked_mul(self.record_size)
            .and_then(|offset| HEADER_SIZE.checked_add(offset))
            .ok_or(StorageError::RecordCountOverflow)?;
        Ok(record_start)
    }
}

/// A candidate entry tracked inside the top-k search heap.
///
/// The [`Ord`] implementation ranks by descending quality: larger distance is
/// "greater", so that a max-heap always exposes the *worst* current candidate
/// at its top and makes eviction straightforward. Ties in distance are broken
/// by larger [`VectorId`], so smaller IDs are retained after eviction.
#[derive(PartialEq)]
struct Candidate {
    distance: f32,
    id: VectorId,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

/// An iterator over all records in a [`FlatVectorStore`].
///
/// Produced by [`FlatVectorStore::iter`]. Each item is a zero-copy
/// [`VectorRecordRef`] borrowing directly from the store's memory map.
pub struct FlatVectorStoreIter<'a> {
    store: &'a FlatVectorStore,
    index: usize,
}

impl<'a> Iterator for FlatVectorStoreIter<'a> {
    type Item = VectorRecordRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let record = self.store.record_ref(self.index).ok().flatten()?;
        self.index += 1;
        Some(record)
    }
}

fn initialize_file(
    file: &File,
    dimension: usize,
    capacity: usize,
    record_size: usize,
) -> Result<(), StorageError> {
    let file_len = file_len_for(capacity, record_size)?;
    file.set_len(file_len)?;

    let mut mmap = map_file(file)?;
    mmap[MAGIC_OFFSET..MAGIC_OFFSET + MAGIC.len()].copy_from_slice(&MAGIC);
    write_u32(&mut mmap, VERSION_OFFSET, VERSION)?;
    write_u32(&mut mmap, DIMENSION_OFFSET, dimension as u32)?;
    write_u64(&mut mmap, LEN_OFFSET, 0)?;
    write_u64(&mut mmap, CAPACITY_OFFSET, capacity as u64)?;
    mmap.flush()?;
    Ok(())
}

fn map_file(file: &File) -> Result<MmapMut, StorageError> {
    if file.metadata()?.len() == 0 {
        return Err(StorageError::InvalidHeader("storage file is empty"));
    }

    let mmap = unsafe { MmapMut::map_mut(file)? };
    Ok(mmap)
}

fn validate_header(mmap: &[u8]) -> Result<(), StorageError> {
    if mmap.len() < HEADER_SIZE {
        return Err(StorageError::InvalidHeader("file smaller than header"));
    }

    if mmap[MAGIC_OFFSET..MAGIC_OFFSET + MAGIC.len()] != MAGIC {
        return Err(StorageError::InvalidHeader("magic mismatch"));
    }

    let version = read_u32(mmap, VERSION_OFFSET)?;
    if version != VERSION {
        return Err(StorageError::InvalidHeader("unsupported version"));
    }

    let dimension = read_u32(mmap, DIMENSION_OFFSET)?;
    if dimension == 0 {
        return Err(StorageError::InvalidHeader("dimension must be greater than zero"));
    }

    Ok(())
}

fn record_size_for(dimension: usize) -> Result<usize, StorageError> {
    VECTOR_ID_SIZE
        .checked_add(dimension.checked_mul(FLOAT_SIZE).ok_or(StorageError::RecordCountOverflow)?)
        .ok_or(StorageError::RecordCountOverflow)
}

fn file_len_for(capacity: usize, record_size: usize) -> Result<u64, StorageError> {
    let body_len = capacity
        .checked_mul(record_size)
        .and_then(|bytes| HEADER_SIZE.checked_add(bytes))
        .ok_or(StorageError::RecordCountOverflow)?;
    Ok(body_len as u64)
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, StorageError> {
    let end = offset + mem::size_of::<u32>();
    let slice = bytes
        .get(offset..end)
        .ok_or(StorageError::InvalidHeader("u32 read out of bounds"))?;
    let mut buffer = [0_u8; 4];
    buffer.copy_from_slice(slice);
    Ok(u32::from_le_bytes(buffer))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, StorageError> {
    let end = offset + mem::size_of::<u64>();
    let slice = bytes
        .get(offset..end)
        .ok_or(StorageError::InvalidHeader("u64 read out of bounds"))?;
    let mut buffer = [0_u8; 8];
    buffer.copy_from_slice(slice);
    Ok(u64::from_le_bytes(buffer))
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<(), StorageError> {
    let end = offset + mem::size_of::<u32>();
    let slice = bytes
        .get_mut(offset..end)
        .ok_or(StorageError::InvalidHeader("u32 write out of bounds"))?;
    slice.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) -> Result<(), StorageError> {
    let end = offset + mem::size_of::<u64>();
    let slice = bytes
        .get_mut(offset..end)
        .ok_or(StorageError::InvalidHeader("u64 write out of bounds"))?;
    slice.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_f32(bytes: &mut [u8], offset: usize, value: f32) -> Result<(), StorageError> {
    let end = offset + FLOAT_SIZE;
    let slice = bytes
        .get_mut(offset..end)
        .ok_or(StorageError::InvalidHeader("f32 write out of bounds"))?;
    slice.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn bytes_as_f32_slice(bytes: &[u8]) -> Result<&[f32], StorageError> {
    if !bytes.len().is_multiple_of(FLOAT_SIZE) {
        return Err(StorageError::CorruptRecordLayout);
    }

    let (prefix, floats, suffix) = unsafe { bytes.align_to::<f32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(StorageError::CorruptRecordLayout);
    }

    Ok(floats)
}