# Data directory

Raw UCI Adult data are intentionally not versioned.

The empirical runner downloads the canonical UCI Adult archive and caches it under `data/cache/adult.zip`, which is gitignored. An optional local archive can be supplied with `--data-path`.

Every source, including a cached or local archive, must match the frozen SHA-256 `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb` before analysis. The generated manifest repeats the accepted hash and duplicate-group diagnostics.
