# LT2212 V19 Assignment 3

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Steinunn Rut Friðriksdóttir

## Additional instructions

I added an additional argparse argument, so gendoc now outputs two seperate files, outputfile_train and
outputfile_test. Outputfile_train always has 80% of the data while outputfile_test has 20% of the data.

## Reporting for Part 4

Tests:                         3gram acc   3gram perp  4gram acc   4gram perp  5gram acc    5gram perp
from line 400 to line 800       0.11111     1.10700     0.1         1.09592     0.1         1.09592
from startline to line 600      0.11111     1.10499     0.11111     1.10499     0.11111     1.10524
from startline to line 4000     0.14286     1.13298     0.14286     1.13298     0.14286     1.13325

Accuracy only shows the few times where the classifier happened to be right and in this case, the number
of classes is so high, the accuracy value is virtually the same everywhere. The number of n-grams seems
to be irrelevant however in this context, while accuracy does seem to go up with more amount of data.
Perplexity is a measurement of how well a probability model predicts a sample. The higher the perplexity,
the worse the model is at predicting the next word. In this example, perplexity becomes lower as the number
of n-grams gets higher. This is not surprising as the model has more context to build its predictions from.
However, perplexity seems to go up as the amount of data increases, which surprised be a bit. I would have
thought that the more data you have, the more accurate the predictions become but that does not seem to be
the case here.
