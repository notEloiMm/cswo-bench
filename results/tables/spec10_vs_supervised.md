|                                                         |   test records |   accuracy |   macro P |   macro R |   macro F1 |   weighted F1 |
|:--------------------------------------------------------|---------------:|-----------:|----------:|----------:|-----------:|--------------:|
| ('habitat', 'distilbert-multilingual', False)           |        117.000 |      0.709 |     0.716 |     0.704 |      0.634 |         0.696 |
| ('habitat', 'distilbert-multilingual', True)            |        117.000 |      0.735 |     0.666 |     0.750 |      0.652 |         0.719 |
| ('habitat', 'majority', False)                          |        117.000 |      0.316 |     0.032 |     0.100 |      0.048 |         0.152 |
| ('habitat', 'qwen2.5-7b (zero-shot llm)', False)        |        117.000 |      0.034 |     0.103 |     0.080 |      0.014 |         0.018 |
| ('habitat', 'qwen2.5-vl-7b (zero-shot vlm)', False)     |        117.000 |      0.060 |     0.063 |     0.096 |      0.038 |         0.053 |
| ('habitat', 'tfidf-char+svm', False)                    |        117.000 |      0.795 |     0.671 |     0.741 |      0.697 |         0.788 |
| ('habitat', 'tfidf-char+svm', True)                     |        117.000 |      0.795 |     0.660 |     0.741 |      0.688 |         0.788 |
| ('habitat', 'tfidf-word+svm', False)                    |        117.000 |      0.795 |     0.714 |     0.779 |      0.725 |         0.785 |
| ('habitat', 'tfidf-word+svm', True)                     |        117.000 |      0.795 |     0.709 |     0.779 |      0.719 |         0.786 |
| ('water_level', 'distilbert-multilingual', False)       |        200.000 |      0.220 |     0.174 |     0.203 |      0.161 |         0.202 |
| ('water_level', 'distilbert-multilingual', True)        |        200.000 |      0.220 |     0.173 |     0.192 |      0.165 |         0.212 |
| ('water_level', 'majority', False)                      |        200.000 |      0.245 |     0.020 |     0.083 |      0.033 |         0.096 |
| ('water_level', 'qwen2.5-7b (zero-shot llm)', False)    |        200.000 |      0.360 |     0.567 |     0.290 |      0.335 |         0.331 |
| ('water_level', 'qwen2.5-vl-7b (zero-shot vlm)', False) |        200.000 |      0.340 |     0.509 |     0.303 |      0.332 |         0.310 |
| ('water_level', 'tfidf-char+svm', False)                |        200.000 |      0.235 |     0.208 |     0.187 |      0.179 |         0.226 |
| ('water_level', 'tfidf-char+svm', True)                 |        200.000 |      0.245 |     0.223 |     0.200 |      0.192 |         0.239 |
| ('water_level', 'tfidf-word+svm', False)                |        200.000 |      0.220 |     0.189 |     0.175 |      0.172 |         0.216 |
| ('water_level', 'tfidf-word+svm', True)                 |        200.000 |      0.225 |     0.206 |     0.189 |      0.184 |         0.223 |