# Installation
The synthyverse is unique in its modular installation set-up. To avoid conflicting dependencies, we provide various installation templates. Each template installs only those dependencies which are required to access certain modules. 

Templates provide installation for specific generators, the evaluation module, and more. Install multiple templates to get access to multiple modules of the synthyverse, e.g., multiple generators and evaluation. 

**We strongly advise to only install templates which you require during a specific run. Installing multiple templates gives rise to potential dependency conflicts. Use separate virtual environments across installations.**

**Note that the core installation without any template doesn't install any modules.**

See the [overview of templates](https://github.com/synthyverse/synthyverse/blob/main/synthyverse/TEMPLATES.md).

### General Installation Template

```bash
pip install synthyverse[template]
```

### Installation Examples
```bash
pip install synthyverse[ctgan]
```


```bash
pip install synthyverse[arf,bn,ctgan,tvae]
```

```bash
pip install synthyverse[ctgan,eval]
```