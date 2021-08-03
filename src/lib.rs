// Deny unsafe code, but allow unsafe code in tests.
#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(clippy::pedantic, clippy::nursery)]

// Re-alias the rayon crate, to allow features to be the name of the crate
#[cfg(feature = "rayon")]
use rayon_crate as rayon;

use std::cmp::Ordering;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::num::NonZeroU64;

use dashmap::setref::multiple::RefMulti;
use dashmap::DashSet;
use murmur3::murmur3_x64_128;
use rand::distributions::{Distribution, Standard};

#[derive(Clone, Debug)]
pub struct NodeSelection<T: Hash + Eq> {
    nodes: DashSet<Node<T>>,
}

impl<T> NodeSelection<T>
where
    T: Hash + Eq,
{
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: DashSet::with_capacity(capacity),
        }
    }

    /// Adds a new node to the selection.
    ///
    /// # Panics
    ///
    /// This method will panic if the provided node has an id that is already
    /// present in the selection. For the non-panicking version of this method,
    /// see [`try_add_node`](#try_add_node).
    #[inline]
    pub fn add_node(&self, node: Node<T>) {
        let id = node.node_id;
        if !self.nodes.insert(node) {
            panic!("Node with duplicate id added: {}", id);
        }
    }

    /// Tries to add the node to the selection.
    ///
    /// # Errors
    ///
    /// This method will fail if the node has an id that is already present in
    /// the selection.
    #[inline]
    pub fn try_add_node(&self, node: Node<T>) -> Result<(), ()> {
        if self.nodes.contains(&node) {
            Err(())
        } else {
            self.nodes.insert(node);
            Ok(())
        }
    }

    #[inline]
    pub fn remove_node(&self, node: &Node<T>) {
        self.nodes.remove(node);
    }

    #[inline]
    pub fn contains(&self, node: &Node<T>) -> bool {
        self.nodes.contains(node)
    }
}

impl<T> NodeSelection<T>
where
    T: Hash + Eq,
{
    /// Finds a node that is responsible for the provided item, filtering out
    /// nodes that refuse to accept the provided tags. This lookup is performed
    /// in serial. Consider [`par_get_node`](#par_get_node) that uses rayon
    /// to parallelize the lookup.
    pub fn get_node(&self, item: &[u8], tags: Option<DashSet<T>>) -> Option<RefMulti<Node<T>>> {
        self.nodes
            .iter()
            .filter(|node| {
                !node.exclusions.iter().any(|exclusions| {
                    tags.as_ref()
                        .map(|set| set.contains(&exclusions))
                        .unwrap_or_default()
                })
            })
            .max_by(|a, b| {
                a.score(item)
                    .partial_cmp(&b.score(item))
                    .unwrap_or(Ordering::Equal)
            })
    }
}

impl<T> NodeSelection<T>
where
    T: Send + Sync + Hash + Eq,
{
    /// Finds a node that is responsible for the provided item, filtering out
    /// nodes that refuse to accept the provided tags. This lookup is performed
    /// in parallel. Requires the `rayon` feature.
    #[cfg(feature = "rayon")]
    pub fn par_get_node(&self, item: &[u8], tags: Option<DashSet<T>>) -> Option<RefMulti<Node<T>>> {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        self.nodes
            .par_iter()
            .filter(|node| {
                !node.exclusions.par_iter().any(|exclusions| {
                    tags.as_ref()
                        .map(|set| set.contains(&exclusions))
                        .unwrap_or_default()
                })
            })
            .max_by(|a, b| {
                a.score(item)
                    .partial_cmp(&b.score(item))
                    .unwrap_or(Ordering::Equal)
            })
    }
}

impl<T> Default for NodeSelection<T>
where
    T: Hash + Eq,
{
    #[inline]
    fn default() -> Self {
        Self {
            nodes: DashSet::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Node<T: Hash + Eq> {
    node_id: NodeId,
    weight: NonZeroU64,
    seed: u32,
    exclusions: DashSet<T>,
}

impl<T> Node<T>
where
    T: Hash + Eq,
{
    #[inline]
    pub fn random(weight: NonZeroU64) -> Self {
        Self {
            node_id: rand::random(),
            seed: rand::random(),
            weight,
            exclusions: DashSet::new(),
        }
    }

    #[inline]
    pub fn with_id(node_id: NodeId, weight: NonZeroU64) -> Self {
        Self {
            node_id,
            seed: rand::random(),
            weight,
            exclusions: DashSet::new(),
        }
    }

    #[inline]
    pub fn new(node_id: NodeId, weight: NonZeroU64, exclusions: DashSet<T>) -> Self {
        Self {
            node_id,
            weight,
            seed: rand::random(),
            exclusions,
        }
    }

    fn score(&self, item: &[u8]) -> f64 {
        let hash = Hash64(murmur3_x64_128(&mut Cursor::new(item), self.seed).unwrap() as u64)
            .as_normalized_float();
        let score = 1.0 / -hash.ln();
        self.weight.get() as f64 * score
    }
}

impl<T: Hash + Eq> Hash for Node<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node_id.hash(state);
    }
}

impl<T: Hash + Eq> PartialEq for Node<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.node_id.eq(&other.node_id)
    }
}

impl<T: Hash + Eq> Eq for Node<T> {}

impl<T: Hash + Eq> PartialOrd for Node<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.node_id.partial_cmp(&other.node_id)
    }
}

impl<T: Hash + Eq> Ord for Node<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.node_id.cmp(&other.node_id)
    }
}

struct Hash64(u64);

impl Hash64 {
    /// Returns a value from [0, 1).
    fn as_normalized_float(&self) -> f64 {
        const FIFTY_THREE_ONES: u64 = u64::MAX >> (64 - 53);
        const FIFTY_THREE_ZEROS: f64 = ((1u64) << 53) as f64;
        (self.0 & FIFTY_THREE_ONES) as f64 / FIFTY_THREE_ZEROS
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(usize);

impl NodeId {
    /// Generates a new node with an ID. This id must be unique, and is used as
    /// the sole determining factor for equality and ordering of nodes.
    #[inline]
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

impl Distribution<NodeId> for Standard {
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> NodeId {
        NodeId(rng.gen())
    }
}

impl Display for NodeId {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod node_selection_no_exclusions {
    use std::collections::BTreeMap;
    use std::iter::FromIterator;

    use super::*;

    #[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
    enum Empty {}

    #[test]
    fn sanity_check_weighted() {
        let node_selection = NodeSelection::<Empty>::new();

        let mut nodes: BTreeMap<_, usize> = unsafe {
            BTreeMap::from_iter([
                (Node::with_id(NodeId(0), NonZeroU64::new_unchecked(100)), 0),
                (Node::with_id(NodeId(1), NonZeroU64::new_unchecked(200)), 0),
                (Node::with_id(NodeId(2), NonZeroU64::new_unchecked(300)), 0),
            ])
        };

        for node in nodes.keys() {
            node_selection.add_node(node.clone());
        }

        for _ in 0..600 {
            let node_selected =
                node_selection.get_node(&(rand::random::<f64>()).to_le_bytes(), None);
            if let Some(node) = node_selected {
                let node = nodes.get_mut(node.key()).unwrap();
                *node += 1;
            }
        }

        for (node, counts) in nodes {
            let anchor = node.weight.get() as usize;
            let range = (anchor - 50)..(anchor + 50);
            assert!(range.contains(&counts));
        }
    }

    #[test]
    fn sanity_check_unweighted() {
        let node_selection = NodeSelection::<Empty>::new();

        let mut nodes: BTreeMap<_, usize> = unsafe {
            BTreeMap::from_iter([
                (Node::with_id(NodeId(0), NonZeroU64::new_unchecked(1)), 0),
                (Node::with_id(NodeId(1), NonZeroU64::new_unchecked(1)), 0),
                (Node::with_id(NodeId(2), NonZeroU64::new_unchecked(1)), 0),
            ])
        };

        for node in nodes.keys() {
            node_selection.add_node(node.clone());
        }

        for _ in 0..600 {
            let node_selected =
                node_selection.get_node(&(rand::random::<f64>()).to_le_bytes(), None);
            if let Some(node) = node_selected {
                let node = nodes.get_mut(node.key()).unwrap();
                *node += 1;
            }
        }

        let range = (200 - 30)..(200 + 30);
        for counts in nodes.values() {
            assert!(range.contains(counts));
        }
    }
}

#[cfg(test)]
mod node_selection_exclusions {
    use std::collections::BTreeMap;
    use std::iter::FromIterator;

    use super::*;

    #[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
    enum Exclusions {
        A,
    }

    #[test]
    fn sanity_check_weighted() {
        let node_selection = NodeSelection::<Exclusions>::new();

        let exclusions = DashSet::from_iter([Exclusions::A]);

        let mut nodes: BTreeMap<_, usize> = unsafe {
            BTreeMap::from_iter([
                (Node::with_id(NodeId(0), NonZeroU64::new_unchecked(100)), 0),
                (
                    Node::new(
                        NodeId(1),
                        NonZeroU64::new_unchecked(200),
                        exclusions.clone(),
                    ),
                    0,
                ),
                (Node::with_id(NodeId(2), NonZeroU64::new_unchecked(300)), 0),
            ])
        };

        for node in nodes.keys() {
            node_selection.add_node(node.clone());
        }

        for _ in 0..600 {
            let node_selected = node_selection.get_node(
                &(rand::random::<f64>()).to_le_bytes(),
                Some(exclusions.clone()),
            );
            if let Some(node) = node_selected {
                let node = nodes.get_mut(node.key()).unwrap();
                *node += 1;
            }
        }

        dbg!(&nodes);

        for (node, counts) in nodes {
            let anchor = node.weight.get() as usize * 3 / 2;
            if node.exclusions.is_empty() {
                let range = (anchor - 50)..(anchor + 50);
                assert!(range.contains(&counts));
            } else {
                assert_eq!(counts, 0);
            }
        }
    }

    #[test]
    fn sanity_check_unweighted() {
        let node_selection = NodeSelection::<Exclusions>::new();

        let exclusions = DashSet::from_iter([Exclusions::A]);

        let mut nodes: BTreeMap<_, usize> = unsafe {
            BTreeMap::from_iter([
                (Node::with_id(NodeId(0), NonZeroU64::new_unchecked(1)), 0),
                (
                    Node::new(NodeId(1), NonZeroU64::new_unchecked(1), exclusions.clone()),
                    0,
                ),
                (Node::with_id(NodeId(2), NonZeroU64::new_unchecked(1)), 0),
            ])
        };

        for node in nodes.keys() {
            node_selection.add_node(node.clone());
        }

        for _ in 0..600 {
            let node_selected = node_selection.get_node(
                &(rand::random::<f64>()).to_le_bytes(),
                Some(exclusions.clone()),
            );
            if let Some(node) = node_selected {
                let node = nodes.get_mut(node.key()).unwrap();
                *node += 1;
            }
        }

        let range = (300 - 30)..(300 + 30);
        for (node, counts) in nodes {
            if node.exclusions.is_empty() {
                assert!(range.contains(&counts));
            } else {
                assert_eq!(counts, 0);
            }
        }
    }
}
