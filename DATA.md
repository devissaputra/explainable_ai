# Dataset Card — UCI Adult

## Source

**UCI Machine Learning Repository, dataset 2: Adult**  
DOI: `10.24432/C5XW20`  
Canonical archive: `https://archive.ics.uci.edu/static/public/2/adult.zip`  
License reported by UCI: **CC BY 4.0**

UCI describes the task as predicting whether annual income exceeds USD 50K and reports 48,842 instances with 14 predictors.

## Frozen archive identity

Expected SHA-256 of `adult.zip`:

`7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`

The runner validates this before parsing. A cached or locally supplied archive is subject to the same check. A different source byte sequence requires an explicit protocol revision.

## Data construction

The runner parses `adult.data` and `adult.test`, removes the documentation line in the test file, normalizes target labels with and without the trailing period, and combines the two files for repeated grouped robustness analysis.

The study does **not** claim to reproduce results based on UCI's original fixed train/test split.

## Outcome

The target indicates whether recorded annual income exceeds USD 50K. This historical benchmark label is not treated as a statement of a person's worth or as a justified real-world decision objective.

## Missing values and preprocessing

Question-mark values are treated as missing. Numeric predictors use median imputation and standardization. Categorical predictors use most-frequent imputation and one-hot encoding. All transformations are fitted within the training pipeline.

## Duplicate-aware split

Exact predictor vectors are hashed into groups before splitting. `StratifiedGroupKFold(n_splits=4, shuffle=True)` keeps each exact predictor group on only one side of a split while approximately preserving target prevalence.

The generated dataset manifest records the number of unique predictor groups, the number of duplicate predictor rows beyond their first occurrence, and the largest exact predictor group. The primary result records train/test exact predictor-group overlap, which must be zero.

## Protected/social attributes

The study identifies `race` and `sex` for a controlled feature-governance comparison. One condition includes them; another removes them before fitting. Held-out race and sex values remain available strictly for descriptive subgroup auditing of both model conditions.

Removing these columns does not remove proxies, structural inequity, selection effects or measurement limitations.

## Limitations

Adult is a historical U.S. census-derived benchmark. The binary income threshold compresses complex socioeconomic circumstances and is not representative of current labor markets, other countries, or a justified automated-decision target.
