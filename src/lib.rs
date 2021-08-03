use std::io::Cursor;
use std::{collections::HashSet, num::NonZeroU64};

use murmur3::murmur3_x64_128;
use rand::distributions::{Distribution, Standard};

#[derive(Clone, Debug)]
pub struct NodeSelection {
    nodes: HashSet<Node>,
}

impl NodeSelection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: HashSet::with_capacity(capacity),
            ..NodeSelection::default()
        }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.insert(node);
    }

    pub fn remove_node(&mut self, node: &Node) {
        self.nodes.remove(node);
    }

    pub fn contains(&self, node: &Node) -> bool {
        self.nodes.contains(node)
    }

    pub fn get_node(&self, item: &[u8]) -> Option<&Node> {
        let mut highest_score = -1f64;
        let mut champion = None;
        for node in &self.nodes {
            let score = node.score(item);
            if score > highest_score {
                champion = Some(node);
                highest_score = score;
            }
        }

        champion
    }
}

impl Default for NodeSelection {
    fn default() -> Self {
        Self {
            nodes: HashSet::new(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Node {
    node_id: NodeId,
    weight: NonZeroU64,
    seed: u32,
}

impl Node {
    pub fn new(weight: NonZeroU64) -> Self {
        Self {
            node_id: rand::random(),
            seed: rand::random(),
            weight,
        }
    }

    pub fn with_id(node_id: NodeId, weight: NonZeroU64) -> Self {
        Self {
            node_id,
            seed: rand::random(),
            weight,
        }
    }

    fn score(&self, item: &[u8]) -> f64 {
        let hash = Hash64(murmur3_x64_128(&mut Cursor::new(item), self.seed).unwrap() as u64)
            .as_normalized_float();
        let score = 1.0 / -hash.ln();
        self.weight.get() as f64 * score
    }
}

struct Hash64(u64);

impl Hash64 {
    /// Returns a value from [0, 1).
    fn as_normalized_float(&self) -> f64 {
        const FIFTY_THREE_ONES: u64 = u64::MAX >> (64 - 53);
        const FIFTY_THREE_ZEROS: f64 = ((1 as u64) << 53) as f64;
        (self.0 & FIFTY_THREE_ONES) as f64 / FIFTY_THREE_ZEROS
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(usize);

impl NodeId {
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}

impl Distribution<NodeId> for Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> NodeId {
        NodeId(rng.gen())
    }
}

#[cfg(test)]
mod node_selection {
    use std::{collections::BTreeMap, iter::FromIterator};

    use super::*;

    #[test]
    fn sanity_check_weighted() {
        let mut node_selection = NodeSelection::new();

        let mut nodes: BTreeMap<Node, usize> = unsafe {
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
            let node_selected = node_selection.get_node(&(rand::random::<f64>()).to_le_bytes());
            if let Some(node) = node_selected {
                let node = nodes.get_mut(node).unwrap();
                *node += 1;
            }
        }

        for (i, counts) in nodes.values().enumerate() {
            let anchor = (i + 1) * 100;
            let range = (anchor - 50)..(anchor + 50);
            assert!(range.contains(counts));
        }
    }

    #[test]
    fn sanity_check_unweighted() {
        let mut node_selection = NodeSelection::new();

        let mut nodes: BTreeMap<Node, usize> = unsafe {
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
            let node_selected = node_selection.get_node(&(rand::random::<f64>()).to_le_bytes());
            if let Some(node) = node_selected {
                let node = nodes.get_mut(node).unwrap();
                *node += 1;
            }
        }

        let range = (200 - 30)..(200 + 30);
        for counts in nodes.values() {
            assert!(range.contains(counts));
        }
    }
}
