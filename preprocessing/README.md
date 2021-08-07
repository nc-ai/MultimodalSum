# Instructions
The preprocessing script is based on [Luigi workflows](https://github.com/spotify/luigi), which parallels preprocessing steps over files to achieve a speed-up. The script is adjusted to work for [Amazon](http://jmcauley.ucsd.edu/data/amazon/links.html) and [Yelp](https://www.yelp.nl/dataset/challenge) data.

Overall, the preprocessing workflow executes the following logic:

1. **Preparation** - groups each product/business reviews to separate CSV files that have proper columns.
2. **Tokenization** - tokenizes sequences with the Moses (reversible) tokenizer.
3. **Subsampling** - filters too short reviews, very unpopular and too popular products/businesses.
4. **Partitioning** - assigns the groups of reviews to training validation partitions.

The output of each step is used as input to a subsequent step, and thus no re-computation is needed of upstream steps if the downstream step needs to be recomputed (e.g., if subsampling parameters change).

## Notes

* Tokenization is performed using a reversible tokenizer Moses, and it is relatively time-consuming to tokenize a large amount of data. Preprocessing can take up to a day on the full Amazon dataset.

* Tokenization is not multi-processing at the moment of writing, which adds to the preprocessing time.

## License

MIT