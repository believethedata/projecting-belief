# Data preprocessing
## Transformations
The notebooks in this folder transform the raw data from each country's census surveys into a standardised format, so that they can be joined together and used for later modelling. They also add some columns that capture age bands and year of birth cohorts. The outputs of this step are stored in the `processed_data` folder.

## Imputations
The England & Wales data for 2021 was censored if religious populations of a given age and sex fell below a threshold size within each LDA. The `EW_preprocessing` notebook imputes theses censored data with a simple approach that ensures the overall England & Wales totals are correct.
