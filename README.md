# _taxus_

### Gaussian Process models for transcriptome data

[![CI](https://github.com/holmrenser/taxus/actions/workflows/ci.yml/badge.svg)](https://github.com/holmrenser/taxus/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/holmrenser/taxus/badge.svg?branch=main)](https://coveralls.io/github/holmrenser/taxus?branch=main)

```
pip install taxus
```

```python
import taxus as tx

gp = tx.GP('~ time + treatment', covariates, counts, kernel='rbf', likelihood='poisson')
elbo = gp.fit()
```
