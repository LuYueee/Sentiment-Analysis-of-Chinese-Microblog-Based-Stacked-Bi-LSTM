# Sentiment Analysis of Chinese Microblog based on Stacked Bidirectional LSTM
In this work, we propose a Chinese sentiment analysis method by incorporating a word2vec model and a stacked bidirectional long short-term memory (namely Stacked Bi-LSTM) model. We first employ the word2vec model to capture semantic features of words and transfer words into high-dimensional word vectors. We evaluate the performance of two typical word2vec models: continuous bag-of-words (CBOW) and skip-gram. We then use the Stacked Bi-LSTM model to conduct the feature extraction of sequential word vectors. We next apply a binary softmax classifier to predict the sentiment orientation by using semantic and contextual features. Moreover, we also conduct extensive experiments on the real dataset collected from Weibo (i.e., one of the most popular Chinese microblogs). 

Please refer to our paper for more technical details. https://ieeexplore.ieee.org/document/8667413

## Cite and Acknowledgement
Please cite our paper when you use this dataset. The main part of algorithm was proposed and implemented by Yue Lu who was an undergraduate student in Faculty of Information Technology, Macau University of Science and Technology and is now a Ph.D. student at University of California, Riverside. 

#### [Plain Text]
```
J. Zhou, Y. Lu, H.-N. Dai, H. Wang and H. Xiao, "Sentiment Analysis of Chinese Microblog Based on Stacked Bidirectional LSTM," in IEEE Access, vol. 7, pp. 38856-38866, 2019.doi: 10.1109/ACCESS.2019.2905048
```

#### [Bibtex Entry]
```
@ARTICLE{YLu:IEEEAccess19, 
	author={Junhao Zhou and Yue Lu and Hong-Ning Dai and Hao Wang and Hong Xiao}, 
	journal={IEEE Access}, 
	title={Sentiment Analysis of Chinese Microblog Based on Stacked Bidirectional LSTM}, 
	year={2019}, 
	volume={7}, 
	number={}, 
	pages={38856-38866}, 
	doi={10.1109/ACCESS.2019.2905048}, 
}
```

## File structure 
- 1 Crawler 
- 2 Preprocessing
- 3 Word representation
- 4 Document representation + Sentiment Ana
- 5 Visualization
- Corpus
- Data
- Model
