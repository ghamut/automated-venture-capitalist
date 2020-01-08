# automated-venture-capitalist
Contains information for the AAAI-KDF'20 paper "The Automated Venture Capitalist".

You can perform analysis presented in the paper by running `Analysis.m` end-to-end.
- `individualData.csv` contains information at the individual-level.
- `teamData.csv` contains information at the team-level (which was featurized from the individual-level data).
- `results/results.csv` contains all model performance with leave-one-team-out cross-validation.
- `modelCoeff/*csv` contains model coefficients of logistic regression model.
