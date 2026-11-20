# Dataset sources and notices

The selected suite data are from **Benchmark Suite for Clustering Algorithms,
v1.2.0**, pinned to commit `4199c2fb21e39bab604106f8589480e0ed758df6`:
<https://github.com/gagolews/clustering-data-v1/tree/4199c2fb21e39bab604106f8589480e0ed758df6>.
The suite describes its datasets as supplied for research purposes unless
otherwise stated. Preserve each dataset's accompanying description, attribution
and applicable terms; these are copied into `notices/`.

The selected datasets are ring_noisy, circles, chainlink, Iris, Ecoli, Yeast,
Glass, Wine, Statlog image segmentation, WDBC, Ionosphere, Sonar, worms_64,
MNIST digits10k and Fashion-MNIST fashion10k. The image datasets are test subsets,
not complete training/test collections. Fashion-MNIST's original MIT notice is
included with its description.

Banknote Authentication is the UCI source at
<https://archive.ics.uci.edu/dataset/267/banknote+authentication>, DOI
`10.24432/C55P57`, attributed to Volker Lohweg (2012). The archived source record
identifies CC BY 4.0. The four features exclude the class column.

The Sonar rank experiment additionally fetches UCI's original archive from
<https://archive.ics.uci.edu/static/public/151/connectionist+bench+sonar+mines+vs+rocks.zip>
to verify the mapping of the suite's labels to rocks and metal cylinders. The
original numerical units are retained for this experiment.

All download URLs, file sizes and SHA256s are in `sources.json`. No package-wide
license statement overrides the terms of an individual dataset. Descriptions in
`notices/` retain the original attributions and dataset-specific references.
