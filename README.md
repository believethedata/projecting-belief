# projecting-belief

This repo contains all the data and code used to produce the [Projecting Belief](https://believethedata.org/just-one-third-of-uk-population-will-be-christian-by-2031/) article. An overview of the techniques used can be found in a companion article [here](https://believethedata.org/projecting-belief-our-model/).

To replicate the results, run the notebooks in the following order:
 1. Process the raw data by running the notebooks in the `preprocessing` folder.
 2. Run the notebook in the `model_selection` folder to derive weights for the ensemble model.
 3. Run the notebooks in the `projections/percentages` and `projections/populations` folders.
 4. Combine these projections by running the notebook in the `projections/combined` folder.
 5. To replicate the chart data from the article, run the notebook in `article_charts`.

Further details on each step can be found in `README.md` files in the respective folders.

Please feel free to email with any questions at jack@believethedata.org.
