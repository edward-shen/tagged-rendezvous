//!

// Deny unsafe code, but allow unsafe code in tests.
#![cfg_attr(not(test), deny(unsafe_code))]
#![warn(
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items
)]
#![deny(missing_docs)]

use dashmap::mapref::one::RefMut;
// Re-alias the rayon crate, to allow features to be the name of the crate
#[cfg(feature = "rayon")]
use rayon_crate as rayon;

use std::cmp::Ordering;
use std::error::Error;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::iter::FromIterator;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

#[cfg(feature = "rayon")]
use rayon::iter::{FromParallelIterator, IntoParallelIterator, ParallelExtend};

use dashmap::mapref::multiple::RefMulti;
use dashmap::{DashMap, DashSet};
use murmur3::murmur3_x64_128;

// Re-export the dashmap crate so users don't need to manually import it
pub use dashmap;

/// Provides a way to consistently select some weighted bucket (or node) given
/// some value.
///
/// The exact implementation is based on the algorithm described in
/// [Schindelhauer and Schomaker "Weighted Distributed Hash Tables"][paper].
/// As a result, modifications of weights (or additions and removals of nodes)
/// can be performed with perfect stability and perfect precision of node
/// weighing; subsequently, only the minimum number of keys are remapped to new
/// nodes.
///
/// This node does not perform any sort of caching. As a result, it's best to
/// use this with some form of cache, whose entries are invalidated when a node
/// is added, removed, or has its weight modified. This is because lookups for
/// a specific node are at best, linear, and at worst, quadratic. To be
/// specific, there is always a linear lookup cost for a given key, determined
/// by the number of nodes present. If nodes utilize tag exclusion, then the
/// cost becomes quadratic to determine if a node should be excluded given a set
/// of tags.
///
/// This implementation attempts to be performant and concurrent as possible,
/// using a [`DashMap`] and [`DashSet`]s to provide concurrent access and
/// `rayon` to provide parallel lookup, if the feature is enabled.
///
/// # Excluding tags
///
/// This is slightly different from a standard weighted hash table as it also
/// implements the ability to allow nodes to exclude themselves from being
/// selected. This is provided via the `ExclusionTag` type, and should be an
/// `enum`.
///
/// Generally speaking, `ExclusionTag`s must at least [`Hash`] and [`Eq`]. This
/// is a requirement of storing the `ExclusionTag` in a Set implementation.
/// However, if using `rayon` to allow for parallel lookup, the trait bounds on
/// `ExclusionTag`s is restricted to types that are also [`Send`] and [`Sync`].
///
/// This feature is asymptotically expensive, as this requires checking all
/// nodes to see if they exclude the given tags. As a result, this should be
/// used with care, and at best the number of exclusions should be relatively
/// small per node.
///
/// Of course, this feature can be disabled if `ExclusionTag` is the unit type,
/// which reduces lookup costs to be linear.
///
/// Excluded nodes do not contribute to the total sum of weights. For example,
/// if three nodes exist with equal weights, and one node is excluded, then the
/// probability of selecting a node is the same as if only two nodes were
/// present.
///
/// In a more complex case, if nodes with weights 1, 2, and 3 are present, and
/// the node with weight 2 is excluded, then the probability of selecting a node
/// becomes 25% and 75% respectively.
///
/// # Examples
///
/// Nodes can be added and removed at any time and subsequent lookups will be
/// immediately reflect the new state.
///
/// ```
/// # use weighted_node_selection::*;
/// # fn main() -> () {
/// # example();
/// # }
/// #
/// # fn example() -> Option<()> {
/// use std::num::NonZeroUsize;
///
/// let selector = NodeSelection::new();
///
/// // Add a node with a weight of 1
/// let node_1 = Node::<(), _>::new(NonZeroUsize::new(1)?, ());
/// selector.add_node(node_1.clone());
///
/// // Lookups will always return the first node, since it's the only one.
/// let looked_up_node = selector.get_node(b"hello world")?;
/// assert_eq!(looked_up_node.value(), &node_1);
/// // Drop the reference ASAP, see `get_node` docs for more info.
/// std::mem::drop(looked_up_node);
///
/// // Add a node with a weight of 2
/// let node_2 = Node::<(), _>::new(NonZeroUsize::new(2)?, ());
/// selector.add_node(node_2.clone());
///
/// // Any caches should be invalidated now, since we changed the state.
/// // Now there's a 33% chance of selecting the first node, and a 67% chance of
/// // selecting the second node.
///
/// // Do something with the node...
///
/// // Now remove the first node
/// selector.remove_node(node_1.id());
///
/// // Any caches should be invalidated now, since we changed the state.
/// // Lookups will now always return the second node, since it's the only one.
///
/// let looked_up_node = selector.get_node(b"hello world")?;
/// assert_eq!(looked_up_node.value(), &node_2);
/// # None
/// # }
/// ```
///
/// Modifying a weight is pretty ergonomic, but remember that caches should be
/// invalidated after modifying the weights.
///
/// ```
/// # use weighted_node_selection::*;
/// # fn main() -> () {
/// # example();
/// # }
/// #
/// # fn example() -> Option<()> {
/// use std::num::NonZeroUsize;
///
/// let selector = NodeSelection::new();
///
/// // Add a node with a weight of 1
/// let node_1 = Node::<(), _>::new(NonZeroUsize::new(1)?, ());
/// selector.add_node(node_1.clone());
///
/// // Add a node with a weight of 2
/// let node_2 = Node::<(), _>::new(NonZeroUsize::new(2)?, ());
/// selector.add_node(node_2.clone());
///
/// // Now there's a 33% chance of selecting the first node, and a 67% chance of
/// // selecting the second node. Lets modify the weight so there's an equal
/// // chance of selecting either node.
///
/// selector.get_node_mut(node_1.id())?.set_weight(NonZeroUsize::new(1)?);
///
/// // Any caches should be invalidated now.
/// # None
/// # }
/// ```
///
/// If exclusion tags are used, then excluded nodes do not contribute to the
/// sum weights of nodes.
///
/// ```
/// # use weighted_node_selection::*;
/// # fn main() -> () {
/// # example();
/// # }
/// #
/// # fn example() -> Option<()> {
/// use std::num::NonZeroUsize;
/// use std::iter::FromIterator;
///
/// use dashmap::DashSet;
///
/// let selector = NodeSelection::new();
///
/// #[derive(Clone, PartialEq, Eq, Hash)]
/// enum ExclusionTag {
///    Foo,
/// }
///
/// // Add a 3 nodes, one with an exclusion tag.
/// let exclusions = DashSet::from_iter([ExclusionTag::Foo]);
/// let node_1 = Node::new(NonZeroUsize::new(1)?, ());
/// selector.add_node(node_1.clone());
/// let node_2 = Node::with_exclusions(NonZeroUsize::new(1)?, (), exclusions.clone());
/// selector.add_node(node_2.clone());
/// let node_3 = Node::new(NonZeroUsize::new(1)?, ());
/// selector.add_node(node_3.clone());
///
/// // There's a 33% chance of selecting any node, if no tags are provided.
/// selector.get_node(b"hello world")?;
///
/// // There's a equal chance of getting node 1 or node 3 since node 2 is
/// // excluded.
/// selector.get_node_with_exclusions(b"hello world", &exclusions)?;
/// # None
/// # }
/// ```
///
/// [paper]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.414.9353
#[derive(Clone, Debug)]
pub struct NodeSelection<ExclusionTags: Hash + Eq, Metadata> {
    /// The list of nodes to select from.
    nodes: DashMap<NodeId, Node<ExclusionTags, Metadata>>,
}

/// A duplicate id was provided. This contains the duplicated ID that was
/// provided.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct DuplicateIdError(NodeId);

impl Display for DuplicateIdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A duplicate ID {} was provided", self.0)
    }
}

impl Error for DuplicateIdError {}

impl<ExclusionTags, Metadata> NodeSelection<ExclusionTags, Metadata>
where
    ExclusionTags: Hash + Eq,
{
    /// Constructs a new empty instance.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructs a new `NodeSelection`, ensuring that at least `capacity`
    /// nodes can be added without re-allocating.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: DashMap::with_capacity(capacity),
        }
    }

    /// Adds a new node to the selection. If lookups are cached, then callers
    /// must invalidate the cache after adding a node. Otherwise, it is possible
    /// to have incorrectly distributed selections.
    ///
    /// # Safety
    ///
    /// This method may deadlock if called when holding any sort of reference
    /// into the map. Callers must ensure that references are dropped as soon as
    /// possible, especially in single threaded contexts.
    ///
    /// # Panics
    ///
    /// This method will panic if the provided node has an id that is already
    /// present in the selection. For the non-panicking version of this method,
    /// see [`try_add_node`](#try_add_node).
    #[inline]
    pub fn add_node(&self, node: Node<ExclusionTags, Metadata>) {
        let id = node.node_id;
        if self.nodes.insert(id, node).is_some() {
            panic!("Node with duplicate id added: {}", id);
        }
    }

    /// Tries to add the node to the selection. If lookups are cached, then
    /// callers must invalidate the cache after adding a node. Otherwise, it is
    /// possible to have incorrectly distributed selections.
    ///
    /// # Safety
    ///
    /// This method may deadlock if called when holding any sort of reference
    /// into the map. Callers must ensure that references are dropped as soon as
    /// possible, especially in single threaded contexts.
    ///
    /// # Errors
    ///
    /// This method will fail if the node has an id that is already present in
    /// the selection.
    #[inline]
    pub fn try_add_node(
        &self,
        node: Node<ExclusionTags, Metadata>,
    ) -> Result<(), DuplicateIdError> {
        if self.nodes.contains_key(&node.id()) {
            Err(DuplicateIdError(node.node_id))
        } else {
            self.nodes.insert(node.id(), node);
            Ok(())
        }
    }

    /// Finds a node that is responsible for the provided item with no
    /// exclusions. This lookup is performed in serial. Consider
    /// [`par_get_node`](#par_get_node) that uses rayon to parallelize the
    /// lookup.
    ///
    /// # Safety
    ///
    /// This method may deadlock if called when holding a mutable reference into
    /// the map. Callers must ensure that references are dropped as soon as
    /// possible, especially in single threaded contexts.
    #[inline]
    #[must_use]
    pub fn get_node(&self, item: &[u8]) -> Option<RefMulti<NodeId, Node<ExclusionTags, Metadata>>> {
        self.get_node_internal(item, None)
    }

    /// Like [`get_node`](#get_node), but allows the caller to provide a list of
    /// tags, which nodes that exclude those tags will be skipped.
    ///
    /// # Safety
    ///
    /// This method may deadlock if called when holding a mutable reference into
    /// the map. Callers must ensure that references are dropped as soon as
    /// possible, especially in single threaded contexts.
    #[inline]
    #[must_use]
    pub fn get_node_with_exclusions(
        &self,
        item: &[u8],
        tags: &DashSet<ExclusionTags>,
    ) -> Option<RefMulti<NodeId, Node<ExclusionTags, Metadata>>> {
        self.get_node_internal(item, Some(tags))
    }

    /// Returns a node that is responsible for the provided item. This lookup
    /// has asymptotic complexity of `O(m * n^2)`, where `m` is the number of
    /// nodes, and `n` is the number of tags given. As a result, callers should
    /// generally use a small number of tags, either on node restrictions or on
    /// the provided tags for best performance.
    ///
    /// # Safety
    ///
    /// This method may deadlock if called when holding a mutable reference into
    /// the map. Callers must ensure that references are dropped as soon as
    /// possible, especially in single threaded contexts.
    fn get_node_internal(
        &self,
        item: &[u8],
        tags: Option<&DashSet<ExclusionTags>>,
    ) -> Option<RefMulti<NodeId, Node<ExclusionTags, Metadata>>> {
        self.nodes
            .iter()
            .filter(|entry| {
                !entry.value().exclusions.iter().any(|exclusions| {
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

    /// Returns a mutable reference to the node given some ID. This is useful
    /// for modifying the weight of a node. If lookups are cached, then callers
    /// must invalidate the cache after adding a node. Otherwise, it is possible
    /// to have incorrectly distributed selections.
    ///
    /// # Safety
    ///
    /// This method may deadlock if called when holding any sort of reference
    /// into the map. Callers must ensure that references are dropped as soon as
    /// possible, especially in single threaded contexts.
    #[inline]
    #[must_use]
    pub fn get_node_mut(
        &self,
        id: NodeId,
    ) -> Option<RefMut<NodeId, Node<ExclusionTags, Metadata>>> {
        self.nodes.get_mut(&id)
    }

    /// Removes a node, returning the node if it existed. If lookups are cached,
    /// then callers must invalidate the cache after successfully removing a
    /// node. Otherwise, it is possible to have incorrectly distributed
    /// selections.
    ///
    /// # Safety
    ///
    /// This method may deadlock if called when holding any sort of reference
    /// into the map. Callers must ensure that references are dropped as soon as
    /// possible, especially in single threaded contexts.
    #[inline]
    #[must_use]
    pub fn remove_node(&self, id: NodeId) -> Option<Node<ExclusionTags, Metadata>> {
        self.nodes.remove(&id).map(|(_id, node)| node)
    }

    /// Returns if the node is selectable from this selector.
    ///
    /// # Safety
    ///
    /// This method may deadlock if called when holding a mutable reference into
    /// the map. Callers must ensure that references are dropped as soon as
    /// possible, especially in single threaded contexts.
    #[inline]
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }
}

impl<ExclusionTags, Metadata> NodeSelection<ExclusionTags, Metadata>
where
    ExclusionTags: Send + Sync + Hash + Eq,
    Metadata: Send + Sync,
{
    /// Finds a node that is responsible for the provided item, filtering out
    /// nodes that refuse to accept the provided tags. This lookup is performed
    /// in parallel. Requires the `rayon` feature.
    #[cfg(feature = "rayon")]
    pub fn par_get_node(
        &self,
        item: &[u8],
        tags: Option<DashSet<ExclusionTags>>,
    ) -> Option<RefMulti<Node<ExclusionTags, Metadata>>> {
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

impl<ExclusionTags, Metadata> Default for NodeSelection<ExclusionTags, Metadata>
where
    ExclusionTags: Hash + Eq,
{
    #[inline]
    fn default() -> Self {
        Self {
            nodes: DashMap::new(),
        }
    }
}

impl<ExclusionTags, Metadata> Extend<(NodeId, Node<ExclusionTags, Metadata>)>
    for NodeSelection<ExclusionTags, Metadata>
where
    ExclusionTags: Hash + Eq,
{
    #[inline]
    fn extend<T: IntoIterator<Item = (NodeId, Node<ExclusionTags, Metadata>)>>(&mut self, iter: T) {
        self.nodes.extend(iter);
    }
}

impl<ExclusionTags, Metadata> FromIterator<(NodeId, Node<ExclusionTags, Metadata>)>
    for NodeSelection<ExclusionTags, Metadata>
where
    ExclusionTags: Hash + Eq,
{
    #[inline]
    fn from_iter<T: IntoIterator<Item = (NodeId, Node<ExclusionTags, Metadata>)>>(iter: T) -> Self {
        Self {
            nodes: DashMap::from_iter(iter),
        }
    }
}

#[cfg(feature = "rayon")]
impl<ExclusionTags, Metadata> FromParallelIterator<Node<ExclusionTags, Metadata>>
    for NodeSelection<ExclusionTags, Metadata>
where
    ExclusionTags: Send + Sync + Hash + Eq,
    Metadata: Send + Sync,
{
    #[inline]
    fn from_par_iter<I>(into_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = Node<ExclusionTags, Metadata>>,
    {
        Self {
            nodes: DashSet::from_par_iter(into_iter),
        }
    }
}

impl<ExclusionTags, Metadata> IntoIterator for NodeSelection<ExclusionTags, Metadata>
where
    ExclusionTags: Hash + Eq,
{
    type Item = (NodeId, Node<ExclusionTags, Metadata>);
    type IntoIter = <DashMap<NodeId, Node<ExclusionTags, Metadata>> as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.nodes.into_iter()
    }
}

#[cfg(feature = "rayon")]
impl<ExclusionTags, Metadata> IntoParallelIterator for NodeSelection<ExclusionTags, Metadata>
where
    ExclusionTags: Send + Sync + Hash + Eq,
    Metadata: Send + Sync,
{
    type Item = <Self as IntoIterator>::Item;
    type Iter = <DashSet<Self::Item> as IntoParallelIterator>::Iter;

    #[inline]
    fn into_par_iter(self) -> <Self as IntoParallelIterator>::Iter {
        self.nodes.into_par_iter()
    }
}

#[cfg(feature = "rayon")]
impl<ExclusionTags, Metadata> ParallelExtend<Node<ExclusionTags, Metadata>>
    for NodeSelection<ExclusionTags, Metadata>
where
    ExclusionTags: Send + Sync + Hash + Eq,
    Metadata: Send + Sync,
{
    #[inline]
    fn par_extend<I>(&mut self, extendable: I)
    where
        I: IntoParallelIterator<Item = Node<ExclusionTags, Metadata>>,
    {
        self.nodes.par_extend(extendable);
    }
}

/// A representation of logical node in some system.
///
/// Each node must contain a weight score, which is used to determine how much
/// traffic a node receives. This score is relative to the all other weights in
/// the node. For example, if you only had nodes with weights of 2 and 4, then
/// the nodes have a 33% and 66% chance of being selected. The important part is
/// the relative ratio of weights, not the absolute values. If it's preferred to
/// have an equal distribution of traffic, then the weights should be equal.
///
/// A node is generic over two types: `ExclusionTags` and `Metadata`.
/// `ExclusionTags` is a type that represents a set of tags that a node declares
/// to not accept. `Metadata` is a type that represents any additional data
/// that a server may need to store about a node, such as labels or a list of
/// IP addresses. The most basic node is where both types are the [`unit`] type.
///
/// Additionally, all nodes must have some unique identifier associated with it.
/// [`NodeId`] is a type that represents this unique identifier, and is only
/// used for [`NodeSelection`] to identify the node.
///
/// # Examples
///
/// The simplest example of a node is a node that accepts all traffic and
/// carries no additional metadata.
///
/// ```
/// # use weighted_node_selection::*;
/// # fn main() -> () {
/// # example();
/// # }
/// #
/// # fn example() -> Option<()> {
/// use std::num::NonZeroUsize;
///
/// // Generates a node with an opaque id and a weight of 4.
/// let node_small = Node::<(), _>::new(NonZeroUsize::new(4)?, ());
/// assert_eq!(*node_small.data(), ());
/// # None
/// # }
/// ```
///
/// Using some metadata, we can get some information that we can use to
/// communicate with the node.
///
/// ```
/// # use weighted_node_selection::*;
/// # fn main() -> () {
/// # example();
/// # }
/// #
/// # fn example() -> Option<()> {
/// use std::net::Ipv4Addr;
/// use std::num::NonZeroUsize;
///
/// // Note that no bounds are on the metadata, but we derive some to make it
/// // easier to test.
/// #[derive(Debug, PartialEq)]
/// struct Metadata {
///    ip_address: Ipv4Addr,
/// }
///
/// // Generates a node with an opaque id, a weight of 4, and with a metadata
/// // object with an IP address.
/// let metadata = Metadata { ip_address: Ipv4Addr::new(10, 0, 0, 1) };
/// let node_small = Node::<(), _>::new(NonZeroUsize::new(4)?, metadata);
/// assert_eq!(node_small.data().ip_address, Ipv4Addr::new(10, 0, 0, 1));
/// # None
/// # }
/// ```
///
/// Finally, you can define a type to use a set of exclusions for a node.
///
/// ```
/// # use weighted_node_selection::*;
/// # fn main() -> () {
/// # example();
/// # }
/// #
/// # fn example() -> Option<()> {
/// use std::iter::FromIterator;
/// use std::net::Ipv4Addr;
/// use std::num::NonZeroUsize;
///
/// use dashmap::DashSet;
///
/// #[derive(PartialEq, Eq, Hash)]
/// enum Exclusions {
///     Foo,
///     Bar,
/// }
///
/// // Generates a node with an opaque id, a weight of 4, and refusing to
/// // accept traffic with the `Foo` tag.
/// let exclusions = DashSet::from_iter([Exclusions::Foo]);
/// let node_small = Node::with_exclusions(NonZeroUsize::new(4)?, (), exclusions);
/// # None
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Node<ExclusionTags: Hash + Eq, Metadata> {
    /// The associated node ID. This should be opaque to library users.
    node_id: NodeId,
    /// The weight of a node. This is a relative value, and is used to determine
    /// how much traffic a node receives.
    weight: NonZeroUsize,
    /// This random seed value is used as a way to differentiate nodes. This is
    /// used to act as a seed value for the hash function, so that two nodes
    /// don't have the same score before weights.
    seed: u32,
    /// A list of items that a node declares to not accept.
    exclusions: DashSet<ExclusionTags>,
    /// Associated data. This is generic, so it can be used for any type,
    /// including `Option<T>`.
    metadata: Metadata,
}

impl<ExclusionTags, Metadata> Node<ExclusionTags, Metadata>
where
    ExclusionTags: Hash + Eq,
{
    /// Constructs a new node with an opaque ID and no exclusions.
    ///
    /// `weight` should not exceed 2^52. `weight` is internally casted as a
    /// [`f64`], and loss of precision may occur if `weight` exceeds [`f64`]'s
    /// mantissa of 52 bits.This should not be a problem for most use cases.
    #[inline]
    #[must_use]
    pub fn new(weight: NonZeroUsize, metadata: Metadata) -> Self {
        Self {
            node_id: NodeId::new_opaque(),
            seed: rand::random(),
            weight,
            exclusions: DashSet::new(),
            metadata,
        }
    }

    /// Constructs a new node with an opaque ID, using the provided metadata and
    /// exclusions.
    ///
    /// `weight` should not exceed 2^52. `weight` is internally casted as a
    /// [`f64`], and loss of precision may occur if `weight` exceeds [`f64`]'s
    /// mantissa of 52 bits.This should not be a problem for most use cases.
    #[inline]
    #[must_use]
    pub fn with_exclusions(
        weight: NonZeroUsize,
        metadata: Metadata,
        exclusions: DashSet<ExclusionTags>,
    ) -> Self {
        Self {
            node_id: NodeId::new_opaque(),
            seed: rand::random(),
            weight,
            exclusions,
            metadata,
        }
    }

    /// Constructs a new node from its components.
    ///
    /// `weight` should not exceed 2^52. `weight` is internally casted as a
    /// [`f64`], and loss of precision may occur if `weight` exceeds [`f64`]'s
    /// mantissa of 52 bits.This should not be a problem for most use cases.
    #[inline]
    #[must_use]
    pub fn from_parts(
        node_id: NodeId,
        weight: NonZeroUsize,
        exclusions: DashSet<ExclusionTags>,
        metadata: Metadata,
    ) -> Self {
        Self {
            node_id,
            weight,
            seed: rand::random(),
            exclusions,
            metadata,
        }
    }

    /// Calculates the score of a node for a given input. This implements
    /// Murmur3 hash alongside highest random weight hashing, also known as
    /// Rendezvous hashing.
    fn score(&self, item: &[u8]) -> f64 {
        // truncation is intended
        #[allow(clippy::cast_possible_truncation)]
        let hash = Hash64(murmur3_x64_128(&mut Cursor::new(item), self.seed).unwrap() as u64)
            .as_normalized_float();
        let score = 1.0 / -hash.ln();
        // This is documented in construction of a node
        #[allow(clippy::cast_precision_loss)]
        {
            self.weight.get() as f64 * score
        }
    }

    /// Sets the weight of the node.
    #[inline]
    pub fn set_weight(&mut self, weight: NonZeroUsize) {
        self.weight = weight;
    }

    /// Fetches the associated data with the node.
    #[inline]
    pub fn data(&self) -> &Metadata {
        &self.metadata
    }

    /// Returns the id of the node.
    #[inline]
    pub fn id(&self) -> NodeId {
        self.node_id
    }
}

impl<ExclusionTags: Hash + Eq, Metadata> Hash for Node<ExclusionTags, Metadata> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node_id.hash(state);
    }
}

impl<ExclusionTags: Hash + Eq, Metadata> PartialEq for Node<ExclusionTags, Metadata> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.node_id.eq(&other.node_id)
    }
}

impl<ExclusionTags: Hash + Eq, Metadata> Eq for Node<ExclusionTags, Metadata> {}

impl<ExclusionTags: Hash + Eq, Metadata> PartialOrd for Node<ExclusionTags, Metadata> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.node_id.partial_cmp(&other.node_id)
    }
}

impl<ExclusionTags: Hash + Eq, Metadata> Ord for Node<ExclusionTags, Metadata> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.node_id.cmp(&other.node_id)
    }
}

/// A representation of a 64 bit hash value.
struct Hash64(u64);

impl Hash64 {
    /// Returns a value from [0, 1).
    fn as_normalized_float(&self) -> f64 {
        /// This is used to mask out the highest 11 bits of a [`f64`].
        const FIFTY_THREE_ONES: u64 = u64::MAX >> (u64::BITS - 53);
        let fifty_three_zeros: f64 = f64::from_bits((1_u64) << 53);
        f64::from_bits(self.0 & FIFTY_THREE_ONES) / fifty_three_zeros
    }
}

/// The counter used for opaque node ids. This is just incrementally updated,
/// which should solve uniqueness issues. If there is a usecase to generate more
/// than [`usize`] IDs, then please file an issue.
static NODE_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// An opaque representation of a node ID.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeId(usize);

impl NodeId {
    /// Generates a new ID from the given number. This id must be unique, and is
    /// used as the sole determining factor for equality and ordering of nodes.
    /// This should only be used when some sort of unique identifier is already
    /// present, or when there needs to be finer control over the internal id
    /// used. Generally, this is not the case, and one should prefer to use
    /// [`new_opaque`](#new_opaque) instead.
    #[inline]
    #[must_use]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Generates a new node with an opaque ID. This should be used and
    /// preferred over [`new`](#new) when the internal node ID is not important.
    #[inline]
    #[must_use]
    pub fn new_opaque() -> Self {
        Self::new(NODE_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed))
    }
}

impl Display for NodeId {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod hash64 {
    use crate::Hash64;

    #[test]
    fn normalized_float() {
        // These numbers are tested against the python implementation
        assert_eq!(
            Hash64(0x1234567890abcdef).as_normalized_float(),
            0.6355555368049276
        );

        assert_eq!(
            Hash64(0xffffffffffffff).as_normalized_float(),
            0.9999999999999999
        );
    }
}

#[cfg(test)]
mod node_selection_no_exclusions {
    use std::collections::BTreeMap;
    use std::iter::FromIterator;

    use super::*;

    #[test]
    fn sanity_check_weighted() {
        let node_selection = NodeSelection::<(), ()>::new();

        let mut nodes: BTreeMap<_, usize> = unsafe {
            BTreeMap::from_iter(
                vec![
                    (NodeId(0), NonZeroUsize::new_unchecked(100)),
                    (NodeId(1), NonZeroUsize::new_unchecked(200)),
                    (NodeId(2), NonZeroUsize::new_unchecked(300)),
                ]
                .into_iter()
                .map(|(id, weight)| (Node::from_parts(id, weight, DashSet::new(), ()), 0)),
            )
        };

        for node in nodes.keys() {
            node_selection.add_node(node.clone());
        }

        for _ in 0..600 {
            let node_selected = node_selection.get_node(&(rand::random::<f64>()).to_le_bytes());
            if let Some(node) = node_selected {
                let node = nodes.get_mut(node.value()).unwrap();
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
        let node_selection = NodeSelection::<(), ()>::new();

        let mut nodes: BTreeMap<_, usize> = unsafe {
            BTreeMap::from_iter(
                vec![
                    (NodeId(0), NonZeroUsize::new_unchecked(1)),
                    (NodeId(1), NonZeroUsize::new_unchecked(1)),
                    (NodeId(2), NonZeroUsize::new_unchecked(1)),
                ]
                .into_iter()
                .map(|(id, weight)| (Node::from_parts(id, weight, DashSet::new(), ()), 0)),
            )
        };

        for node in nodes.keys() {
            node_selection.add_node(node.clone());
        }

        for _ in 0..600 {
            let node_selected = node_selection.get_node(&(rand::random::<f64>()).to_le_bytes());
            if let Some(node) = node_selected {
                let node = nodes.get_mut(node.value()).unwrap();
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
        let node_selection = NodeSelection::<Exclusions, ()>::new();

        let exclusions = DashSet::from_iter([Exclusions::A]);

        let mut nodes: BTreeMap<_, usize> = unsafe {
            BTreeMap::from_iter([
                (
                    Node::from_parts(
                        NodeId(0),
                        NonZeroUsize::new_unchecked(100),
                        DashSet::new(),
                        (),
                    ),
                    0,
                ),
                (
                    Node::from_parts(
                        NodeId(1),
                        NonZeroUsize::new_unchecked(200),
                        exclusions.clone(),
                        (),
                    ),
                    0,
                ),
                (
                    Node::from_parts(
                        NodeId(2),
                        NonZeroUsize::new_unchecked(300),
                        DashSet::new(),
                        (),
                    ),
                    0,
                ),
            ])
        };

        for node in nodes.keys() {
            node_selection.add_node(node.clone());
        }

        for _ in 0..600 {
            let node_selected = node_selection
                .get_node_with_exclusions(&(rand::random::<f64>()).to_le_bytes(), &exclusions);
            if let Some(node) = node_selected {
                let node = nodes.get_mut(node.value()).unwrap();
                *node += 1;
            }
        }

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
        let node_selection = NodeSelection::<Exclusions, ()>::new();

        let exclusions = DashSet::from_iter([Exclusions::A]);

        let mut nodes: BTreeMap<_, usize> = unsafe {
            BTreeMap::from_iter([
                (
                    Node::from_parts(
                        NodeId(0),
                        NonZeroUsize::new_unchecked(1),
                        DashSet::new(),
                        (),
                    ),
                    0,
                ),
                (
                    Node::from_parts(
                        NodeId(1),
                        NonZeroUsize::new_unchecked(1),
                        exclusions.clone(),
                        (),
                    ),
                    0,
                ),
                (
                    Node::from_parts(
                        NodeId(2),
                        NonZeroUsize::new_unchecked(1),
                        DashSet::new(),
                        (),
                    ),
                    0,
                ),
            ])
        };

        for node in nodes.keys() {
            node_selection.add_node(node.clone());
        }

        for _ in 0..600 {
            let node_selected = node_selection
                .get_node_with_exclusions(&(rand::random::<f64>()).to_le_bytes(), &exclusions);
            if let Some(node) = node_selected {
                let node = nodes.get_mut(node.value()).unwrap();
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
