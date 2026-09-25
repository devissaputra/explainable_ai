# Dataset Card — UCI Adult

## Source
UCI Machine Learning Repository, dataset 2: **Adult**  
DOI: https://doi.org/10.24432/C5XW20  
Dataset page: https://archive.ics.uci.edu/dataset/2/adult  
License reported by UCI: CC BY 4.0.

## Scale
UCI reports 48,842 records and 14 predictors derived from census data.

## Outcome
The target indicates whether reported annual income exceeds USD 50K. The runner normalizes both forms with and without a trailing period.

## Missing values
The runner converts `?` and surrounding whitespace to missing values. Numerical columns use median imputation; categorical columns use most-frequent imputation.

## Protected attributes
The experiment explicitly identifies `race` and `sex` as protected/social attributes for a feature-governance comparison. One condition includes them because the study asks how the historical classifier depends on them; the second excludes them. Neither condition is presented as a deployment recommendation.

## Split
Fixed stratified 75/25 train/test split, seed 42. All preprocessing and model fitting happen inside the training pipeline. Permutation importance is calculated on the held-out test data.

## Limitations
Adult is an old US census-derived benchmark. Income threshold classification compresses socioeconomic outcomes into a binary label and cannot support broad claims about people, worth, opportunity or current labor markets.
