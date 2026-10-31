|                                                   |   test records |   accuracy |   macro P |   macro R |   macro F1 |   weighted F1 |
|:--------------------------------------------------|---------------:|-----------:|----------:|----------:|-----------:|--------------:|
| ('habitat', 'distilbert-multilingual', False)     |        117.000 |      0.709 |     0.716 |     0.704 |      0.634 |         0.696 |
| ('habitat', 'distilbert-multilingual', True)      |        117.000 |      0.735 |     0.666 |     0.750 |      0.652 |         0.719 |
| ('habitat', 'majority', False)                    |        117.000 |      0.316 |     0.032 |     0.100 |      0.048 |         0.152 |
| ('habitat', 'tfidf-char+svm', False)              |        117.000 |      0.795 |     0.671 |     0.741 |      0.697 |         0.788 |
| ('habitat', 'tfidf-char+svm', True)               |        117.000 |      0.795 |     0.660 |     0.741 |      0.688 |         0.788 |
| ('habitat', 'tfidf-word+svm', False)              |        117.000 |      0.795 |     0.714 |     0.779 |      0.725 |         0.785 |
| ('habitat', 'tfidf-word+svm', True)               |        117.000 |      0.795 |     0.709 |     0.779 |      0.719 |         0.786 |
| ('water_level', 'distilbert-multilingual', False) |        578.000 |      0.206 |     0.176 |     0.191 |      0.152 |         0.190 |
| ('water_level', 'distilbert-multilingual', True)  |        578.000 |      0.194 |     0.166 |     0.174 |      0.149 |         0.184 |
| ('water_level', 'majority', False)                |        578.000 |      0.292 |     0.024 |     0.083 |      0.038 |         0.132 |
| ('water_level', 'tfidf-char+svm', False)          |        578.000 |      0.265 |     0.251 |     0.216 |      0.216 |         0.263 |
| ('water_level', 'tfidf-char+svm', True)           |        578.000 |      0.261 |     0.240 |     0.216 |      0.209 |         0.261 |
| ('water_level', 'tfidf-word+svm', False)          |        578.000 |      0.235 |     0.211 |     0.208 |      0.195 |         0.236 |
| ('water_level', 'tfidf-word+svm', True)           |        578.000 |      0.244 |     0.230 |     0.215 |      0.207 |         0.247 |