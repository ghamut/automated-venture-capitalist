# automated-venture-capitalist
Contains information for the AAAI-KDF'20 paper "The Automated Venture Capitalist: Data and Methods to Predict the Fate of Startup Ventures".

- `Analysis.m` is a Matlab script that contains the analyses presented in the paper. 

- `data/`
  - `individualData.csv` contains information at the individual-level.
  - `teamData.csv` contains information at the team-level (which was featurized from the individual-level data).

- `results/` 
  - `results.csv` contains all model performance with leave-one-team-out cross-validation.
  - `comparison.csv` contains classification performance comparisons between crowd, judges, and models (Table 2 in paper).

- `modelCoeff/` 
  - `{nomination,success}_coeff.csv` contains logistic regression model coefficients (Table 3 in paper).
  - `{nomination,success}_perf.csv`contains logistic regression model performance when trained on all data.
  
- `functions/` contains supporting scripts.

### Citation
```
@inproceedings{gham2020autovc,
  title={The Automated Venture Capitalist: 
            Data and Methods to Predict the Fate of Startup Ventures},
  author={Ghassemi, Mohammad M. and Song, Christopher and Alhanai, Tuka},
  booktitle={KDF at the Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
