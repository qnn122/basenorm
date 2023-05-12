# basenorm

A baseline method using BERT for entity normalization.

# Need dependencies:
- nltk (conda)
- scipy (conda)
- pronto (conda install -c bioconda pronto)
- pytorch (conda install pytorch -c pytorch)

# To run W2V basenorm.py method:
- tensorflow (conda)
- gensim (conda)

# Training
```
python train.py --do-train
```

If you plan on using the base method made using W2V, add the embeddings file in ./basenorm.
- Gensim Word2Vec embeddings file downloadable [here](http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin)

# Files
- **basenorm.py** : Base method using W2V embeddings with linear layer
- **basenorm_bert_linear.py** : Embeddings obtained using BERT instead of W2V. Linera layer. \
    `score_BB4_onDev: 0.42131147540983604` (n=1) \
    `time to completion ~3 min`
- **basenorm_bert_finetuned.py**: Embeddings obtained using a fine tuned BERT instead of W2V. \
    `score_BB4_onDev: ~0.0` (n=1, epochs = 50) \
    `time to completion ~45 min`
- **basenorm_bert_finetuned_linear.py** :  Embeddings obtained using a fine tuned BERT instead of W2V followed by a linear layer. \
    `score_BB4_onDev: 0.0069` (n=2, epochs = 50) \
    `time to completion ~45 min`

# Licence
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
