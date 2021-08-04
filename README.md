# Tagged Rendezvous

[![unsafe forbidden][unsafe-svg]][unsafe-url] [![codecov][cov-svg]][cov-url] [![docs.rs][docs-svg]][docs-url]

[unsafe-svg]: https://img.shields.io/badge/unsafe-forbidden-success.svg
[unsafe-url]: https://github.com/rust-secure-code/safety-dance
[cov-svg]: https://codecov.io/gh/edward-shen/tagged-rendezvous/branch/master/graph/badge.svg?token=GI53X8LB0R
[cov-url]: https://codecov.io/gh/edward-shen/tagged-rendezvous
[docs-svg]: https://img.shields.io/docsrs/tagged-rendezvous
[docs-url]: https://docs.rs/tagged-rendezvous

`tagged-rendezvous` is a toy crate for implementing [rendezvous hashing] with
the ability to exclude certain nodes based on a generic discriminant. This
crate utilizes the algorithm described in
[Schindelhauer and Schomaker "Weighted Distributed Hash Tables"][paper] to
provide perfect stability and weight precision in the presence of mutations.

[rendezvous hashing]: https://en.wikipedia.org/wiki/Rendezvous_hashing
[paper]: https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.414.9353

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