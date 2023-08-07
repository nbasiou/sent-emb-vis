# sent-emb-vis
Sentence Embeddings Extraction and Visualization

<<<<<<< Updated upstream
This repository implements the computation of Sentence Bert Embeddings for input sentences using the Huggingface API. The code uses the [GLUE Semantic Textual Similarity Benchmark (STSB)](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) dataset. For the sentence embeddings the sentence-transformers model [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is used.

The embeddings are visualized in 3D using the [UMAP transform](https://umap-learn.readthedocs.io/en/latest/) where sentences that come from the same pair are indicated with the same color in the plot.

## Dependencies
Install all the necessary package requirements.

````python
pip install -r requirements.txt
````

## Notes
The code is test it in Python 3.9.17.

You may want to add the following on the top of your code if you get any `numba` related warnings. 
=======

This repository implements the computation of Sentence Bert Embeddings for input sentences using the Huggingface API. The code uses the [GLUE Semantic Textual Similarity Benchmark (STSB)[(http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)] dataset. For the sentence embeddings the sentence-transformers model [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is used.

The embeddings are visualized in 3D using the [UMAP transform](https://umap-learn.readthedocs.io/en/latest/) where sentences that come from the same pair are indicated with the same color in the plot.




## Comments
You may want to add the following on the top of your code if you get any numba related warnings. 
>>>>>>> Stashed changes

````python
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
````

Similarly, if you get a warning from `huggingface/tokenizers` regarding paralellism:

````python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
<<<<<<< Updated upstream
````
=======
````
>>>>>>> Stashed changes
