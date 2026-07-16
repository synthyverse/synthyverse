# Third-party notices

The base `synthyverse` package is distributed under the MIT License. Some
implementations are based on third-party projects and retain the upstream
license obligations listed below.

## MIT-licensed adapted implementations

The MIT license text for these adapted implementations is kept in
`LICENSES/MIT.txt`.

### ARFGenerator

- Synthyverse files: `synthyverse/generators/arf_generator/*`
- Upstream project: `arfpy (https://github.com/bips-hb/arfpy/)`
- Upstream license: MIT License
- Upstream copyright notice: `Copyright (c) 2023 Kristin Blesch, Marvin Wright`
- Imported or last compared against: `Commit 8b63c1b`
- Modifications: `Fallback when nodes contain single value.`
- Upstream NOTICE text, if any: `None`

### CDTDGenerator

- Synthyverse files: `synthyverse/generators/cdtd_generator/*`
- Upstream project: `CDTD (https://github.com/muellermarkus/cdtd)`
- Upstream license: MIT License
- Upstream copyright notice: `Copyright (c) 2025 Markus Mueller`
- Imported or last compared against: `Commit 5bfab87`
- Modifications: `Refactored API.`
- Upstream NOTICE text, if any: `None`

### SMOTEGenerator

- Synthyverse files: `synthyverse/generators/smote_generator/*`
- Upstream project: `CDTD (https://github.com/muellermarkus/cdtd)`
- Upstream license: MIT License
- Upstream copyright notice: `Copyright (c) 2025 Markus Mueller`
- Imported or last compared against: `Commit 5bfab87`
- Modifications: `Refactored API and fallback behaviour for small class sizes.`
- Upstream NOTICE text, if any: `None`

### TabDiffGenerator

- Synthyverse files: `synthyverse/generators/tabdiff_generator/*`
- Upstream project: `TabDiff (https://github.com/MinkaiXu/TabDiff/)`
- Upstream license: MIT License
- Upstream copyright notice: `Copyright 2024 Minkai Xu`
- Imported or last compared against: `Commit 5ecdb33`
- Modifications: `Refactored API, speed-up masked diffusion by vectorizing loops, fast tensor dataloader.`
- Upstream NOTICE text, if any: `None`

### TabCascadeGenerator

- Synthyverse files: `synthyverse/generators/tabcascade_generator/*`
- Upstream project: `TabCascade (https://github.com/muellermarkus/tabcascade)`
- Upstream license: MIT License
- Upstream copyright notice: `Copyright (c) 2026 Markus Mueller`
- Imported or last compared against: `Commit c8f44bb`
- Modifications: `Refactored API, simple ordinal encoding for categories.`
- Upstream NOTICE text, if any: `None`

## Apache-2.0-licensed adapted implementations

The Apache-2.0 license text for these adapted implementations is kept in
`LICENSES/Apache-2.0.txt`.

### TabDDPMGenerator

- Synthyverse files: `synthyverse/generators/tabddpm_generator/*`
- Upstream project: `synthcity (https://github.com/vanderschaarlab/synthcity)`
- Upstream license: Apache License, Version 2.0
- Upstream copyright notice: `Copyright vanderschaarlab 2023`
- Imported or last compared against: `Commit 23f322f`
- Modifications: `Refactored API, allow manual specification of categorical features, fast tensor dataloader.`
- Upstream NOTICE text, if any: `None`

### TabSynGenerator

- Synthyverse files: `synthyverse/generators/tabsyn_generator/*`
- Upstream project: `TabSyn (https://github.com/amazon-science/tabsyn/)`
- Upstream license: Apache License, Version 2.0
- Upstream copyright notice: `Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.`
- Imported or last compared against: `Commit cb5ac0f`
- Modifications: `Refactor API, fast tensor dataloader.`
- Upstream NOTICE text, if any: `Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.`

### AlphaPrecisionBetaRecall fidelity metric

- Synthyverse files: `synthyverse/evaluation/fidelity.py`
- Upstream project: `synthcity (https://github.com/vanderschaarlab/synthcity)`
- Upstream license: Apache License, Version 2.0
- Upstream copyright notice: `Copyright vanderschaarlab 2023`
- Imported or last compared against: `Commit 23f322f`
- Modifications: `Refactored API, allow manual specification k in Beta-Recall.`
- Upstream NOTICE text, if any: `None`


## Business Source License dependency

`CTGANGenerator` and `TVAEGenerator` require the optional third-party `ctgan`
package. That package is distributed under the Business Source License 1.1,
not under the synthyverse MIT License.

- Synthyverse files: `synthyverse/generators/ctgan_generator/*`,
  `synthyverse/generators/tvae_generator/*`
- Upstream project: `ctgan (https://github.com/sdv-dev/CTGAN/)`
- Upstream license: Business Source License 1.1
- Upstream copyright notice: `The Licensed Work is (c) DataCebo, Inc.; license text copyright (c) 2017 MariaDB Corporation Ab, All Rights Reserved.`
- License text: `LICENSES/CTGAN-BSL-1.1.txt`
- Imported or last compared against: `version 0.12.0`
- Modifications: `No ctgan source code is vendored.`
- Upstream NOTICE text, if any: `The Business Source License (this document, or the "License") is not an Open Source license. However, the Licensed Work will eventually be made available under an Open Source License, as stated in this License.`

## Non-vendored dependencies

External runtime dependencies declared by `synthyverse` are not vendored in
this distribution. Those dependencies are distributed separately under their
own license terms.
