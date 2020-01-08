# The Automated Venture Capitalist
Investing is hard because there are an incredible number of factors that influence to the success or failure of ventures. Some of these factors are within a venture's control, and others are not. This respository contains an AI method, tool, and dataset to understand these factors and the complex interactions between them so that organizations and members of the general public can do a better job assessing risk and identifying investment opportunities.

### In this Repository
Specifically, this repository supports information presented in the AAAI-KDF'20 paper "[The Automated Venture Capitalist: Data and Methods to Predict the Fate of Startup Ventures](https://ghassemi.xyz/static/documents/Ghassemi_AAAI_The-automated-venture-capitalist_2020.pdf)".

- `Analysis.m` is a Matlab script that contains the analyses presented in the paper. 

- `data/`
  - `individualData.csv` contains information at the individual-level.
  - `teamData.csv` contains information at the team-level (which was featurized from the individual-level data).

- `results/` 
  - `results.csv` contains all model performance with leave-one-team-out cross-validation.
  - `comparison.csv` contains classification performance comparisons between crowd, judges, and models (Table 2 in paper).

- `modelCoeff/` 
  - `{nomination,success}Model_coeff.csv` contains logistic regression model coefficients (Table 3 in paper).
  - `{nomination,success}Model_perf.csv`contains logistic regression model performance when trained on all data.
  
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

### Contact
If you find this interesting and would like to learn more, please contact [research@ghamut.com](research@ghamut.com)
