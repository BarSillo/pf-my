# RLHedge

A reinforcement learning package for financial hedging applications.

## Installation

Install in development mode:

```bash
pip install -e .
```

## Documentation

working files are located:
```txt
./fin-halp-rl-pub/Week2/dp_qlbs_oneset_m3_ex2_v3.ipynb
./fin-halp-rl-pub/Week2/dp_qlbs_oneset_m3_ex2_v3_upd.ipynb
./fin-halp-rl-run25/notebooks/lab-rl-01.ipynb
./halp-rl-materials-labs/ex/dp_qlbs_oneset_m3_ex2_v3.ipynb

Original locations are downloaded at halp-rl-materials-labs\ex

## Submodules

```sh
# list submodules
$ ls -1d */.git
fin-halp-rl-pub/.git/
halp-fin-ml-3/.git/
ml-fin-halperin/.git/

# 1. Remove the subfolder from git tracking (if it's tracked)
git rm -r --cached path/to/subfolder

# 2. Add it as a submodule
git submodule add <repository-url> path/to/subfolder

# 3. Commit the changes
git add .gitmodules path/to/subfolder
git commit -m "Added subfolder as submodule"
```



```
Mathematical formulas and implementations are documented in the `docs/` directory:
- [GBM Formula](docs/gbm_formula.md)
- [Other Formulas](docs/formulas.md)

## License

MIT License
