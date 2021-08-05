# Tagged Rendezvous

[![unsafe forbidden][unsafe-svg]][unsafe-url] [![codecov][cov-svg]][cov-url] [![docs.rs][docs-svg]][docs-url]

[unsafe-svg]: https://img.shields.io/badge/unsafe-forbidden-success.svg
[unsafe-url]: https://github.com/rust-secure-code/safety-dance
[cov-svg]: https://codecov.io/gh/edward-shen/tagged-rendezvous/branch/master/graph/badge.svg?token=GI53X8LB0R
[cov-url]: https://codecov.io/gh/edward-shen/tagged-rendezvous
[docs-svg]: https://docs.rs/tagged-rendezvous/badge.svg
[docs-url]: https://docs.rs/tagged-rendezvous

`tagged-rendezvous` is a toy crate for implementing [rendezvous hashing] with
the ability to exclude certain nodes based on a generic discriminant. This
crate utilizes the algorithm described in
[Schindelhauer and Schomaker "Weighted Distributed Hash Tables"][paper] to
provide perfect stability and weight precision in the presence of mutations.

The intended use case of this crate is to provide an ergonomic way to
load-balance to potentially weighted nodes for some arbitrary input, while
allowing large networks to respect [geopolitical issues] that often occur with
global networks. In other words, it allows for nodes to provide a discriminant
to exclude themselves from being selected given the need to load balance some
content that may be ethically, morally, or legally impermissible.

This is useful in multiple contexts. For example, a basic CDN architecture may
use this to distribute multiple nodes across multiple countries and ensure that
some content will not appear in nodes where the content is considered illegal.
Another example would be in scenarios of volunteer nodes for some CDN to support
some network as a whole but refuse to serve certain content that is illegal in
their country or content they morally object to.

[rendezvous hashing]: https://en.wikipedia.org/wiki/Rendezvous_hashing
[paper]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.414.9353
[geopolitical issues]: https://en.wikipedia.org/wiki/Layer_8

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

`SPDX-License-Identifier: MIT OR Apache-2.0`