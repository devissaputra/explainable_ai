# Data directory

Raw UCI Adult data are intentionally not versioned.

The empirical runner downloads the canonical UCI Adult archive and caches it under `data/cache/adult.zip`, which is gitignored. An optional local archive can be supplied with `--data-path`.

Every generated empirical manifest records the SHA-256 hash of the exact archive used.
