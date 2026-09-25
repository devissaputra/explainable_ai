# Dataset Card — UCI Adult

## Source

UCI Machine Learning Repository, dataset 2: **Adult**  
DOI: https://doi.org/10.24432/C5XW20  
Dataset page: https://archive.ics.uci.edu/dataset/2/adult  
Canonical archive: https://archive.ics.uci.edu/static/public/2/adult.zip  
License reported by UCI: **CC BY 4.0**.

## Data used

The runner combines `adult.data` and `adult.test`, removes the documentation/header line from the test file, and normalizes target labels with and without the trailing period. The generated manifest stores the SHA-256 hash of the exact archive used.

## Outcome

The target indicates whether the recorded annual income category exceeds USD 50K. This historical binary label is used as a benchmark outcome, not as a statement of a person's worth or opportunity.

## Missing values and preprocessing

Question-mark values are treated as missing. Numeric columns use median imputation and standardization. Categorical columns use most-frequent imputation and one-hot encoding. All preprocessing is fitted inside the training pipeline.

## Protected/social attributes

The study explicitly identifies `race` and `sex` for a controlled feature-governance comparison. One condition includes them to measure model dependence. A second condition removes them before fitting. Held-out group labels are still available for descriptive subgroup auditing of both conditions.

Removing these columns does not remove proxies, structural inequity, selection effects or measurement limitations.

## Split

The primary split is stratified 75/25 with seed 42. Robustness uses four additional fixed seeds: 13, 29, 73 and 101. Explanations are calculated only on held-out observations.

## Limitations

Adult is an older US census-derived benchmark. The binary income threshold compresses complex socioeconomic circumstances and is not representative of current labor markets, other countries, or a justified real-world automated decision by default.
