# automated-venture-capitalist
Contains information for the AAAI-KDF'20 paper "The Automated Venture Capitalist".

You can perform analysis presented in the paper by running `Analysis.m` end-to-end.
- `individualData.csv` contains information at the individual-level.
- `teamData.csv` contains information at the team-level (which was featurized from the individual-level data).
- `results/results.csv` 
  - `results.csv` contains all model performance with leave-one-team-out cross-validation.
  - `comparison.csv` contains classification performance comparisons between crowd, judges, and models (Table 2 in paper).
- `modelCoeff/*csv` 
  - `{nomination,success}_coeff.csv` contains logistic regression model coefficients (Table 3 in paper).
