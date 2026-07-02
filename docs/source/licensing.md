# Licensing

The base `synthyverse` package is distributed under the MIT License.

Some implementations are based on third-party projects. Package-level
third-party attribution, license, NOTICE, and modification details are kept in
`THIRD_PARTY_NOTICES.md`. The third-party license texts are distributed in
`LICENSES/`.

The base install does not depend on the `ctgan` package and does not enable
`CTGANGenerator` or `TVAEGenerator`:

```bash
pip install synthyverse
```

To use CTGAN or TVAE, install the optional extra:

```bash
pip install "synthyverse[ctgan]"
```

That extra installs the third-party `ctgan` package. `ctgan` is distributed
under the Business Source License, not the MIT License. Review the `ctgan`
license terms before installing or using `CTGANGenerator` or `TVAEGenerator`,
especially for service or production use.

The CTGAN license text is kept in `LICENSES/CTGAN-BSL-1.1.txt` for reference.
