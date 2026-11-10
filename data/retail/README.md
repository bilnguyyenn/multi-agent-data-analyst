# UCI Online Retail slices

Chen, D. (2015). Online Retail. UCI Machine Learning Repository.
https://doi.org/10.24432/C5BW33 · CC BY 4.0.

The source archive and streaming conversion are pinned in SOURCE.json. These are
complete seven-day calendar slices in November 2011. All source rows in each
window, duplicates, missing values, cancellations, and unusual signs are retained.
Timestamps are naive source-local values; integer identifiers are preserved.
Development and refresh precede the two evaluation windows. This is a public
temporal development/evaluation split, not a claim of model-pretraining novelty.
The full 541,909-row conversion remains under ignored .retail_source/full.
Reproduce with `python -m analyst retail-import --download`.
