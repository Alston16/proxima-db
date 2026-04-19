use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::distance;
use crate::DistanceMetric;

/// Selects the `k` nearest candidates from `(id, vector)` pairs.
///
/// Results are sorted by ascending distance with deterministic tie-breaking on
/// ascending ID.
pub fn select_topk_by_vector<'a, Id, I>(
    query: &[f32],
    items: I,
    k: usize,
    metric: DistanceMetric,
) -> Vec<(Id, f32)>
where
    Id: Copy + Ord,
    I: IntoIterator<Item = (Id, &'a [f32])>,
{
    if k == 0 {
        return Vec::new();
    }

    let mut heap: BinaryHeap<Candidate<Id>> = BinaryHeap::with_capacity(k + 1);

    for (id, vector) in items {
        let distance = match metric {
            DistanceMetric::L2 => distance::l2_distance(query, vector),
            DistanceMetric::Cosine => distance::cosine_distance(query, vector),
        };
        let candidate = Candidate { id, distance };
        if heap.len() < k {
            heap.push(candidate);
        } else if let Some(worst) = heap.peek() && candidate < *worst {
            heap.pop();
            heap.push(candidate);
        }
    }

    let mut selected: Vec<(Id, f32)> = heap.into_iter().map(|c| (c.id, c.distance)).collect();
    selected.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    selected
}

#[derive(Clone, Copy)]
struct Candidate<Id> {
    id: Id,
    distance: f32,
}

impl<Id: Ord> PartialEq for Candidate<Id> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.distance.total_cmp(&other.distance) == Ordering::Equal
    }
}

impl<Id: Ord> Eq for Candidate<Id> {}

impl<Id: Ord> PartialOrd for Candidate<Id> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Id: Ord> Ord for Candidate<Id> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}