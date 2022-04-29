<center> <h2>Spoiling Fairytales: Learning Outcomes of <em>The Pied Piper</em> </h2>

Brandon Novak

DA 245: Introduction to Cultural Analytics

May 4th, 2022

 </center>

### Introduction

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For the final, we decided to research the story-telling of *The Pied Piper of Hamelin*, a German fairytale. The general story of *The Pied Piper* is that the town of Hamelin in 1284 had a rat infestation and a man with a pipe who was wearing a "pied" pattern clothing offered his services in exchange for monetary compensation. The Piper played his pipe and all the rats in the town appeared in front of the Piper, and he led them into the river to drown. When he returned to Hamelin, the leaders of the town refused to pay him despite agreeing to the deal. As punishment, the Piper started playing his flute, and all the children in the town, similar to the rats, began to follow the Piper. The Piper led the 130 children into the forest, and were never seen again. Here is a [link](https://sites.pitt.edu/~dash/hameln.html#grimm245) of the Grimm's version of the story.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The compelling fact about *The Pied Piper of Hamelin* is that the town of Hamelin claims that the fairy tale is true. In fact, there are historical records such as personal accounts and local German newspapers/chronicles which report on the disappearance of the 130 children. Furthermore, there are many accounts that the local church had a stained glass window of the Pied Piper leading children into the forest as stated in the legend (Dirckx 1980). The fairy tale has been retold and rewritten since the time of the event itself in 1284. Many different cultures have rewritten the story in the setting of their towns. As a result of the retelling of the story, a couple key aspects have changed along the way. The biggest being if the children were actually led away from the town. Many versions claim that the livestock was led astray. While either event is devastating for a town, the stories claiming the children were abducted is a darker tone than the town's livestock being stolen.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this research, we gathered 22 different versions of *The Pied Piper* which were written by many cultures, but mainly dominated by German writers. 12 of the versions tell the story without the abduction of the children meanwhile the remaining 10 versions do mention the stolen children. We use 3 models to predict the outcome of the story. The outcome in which they predict is whether or not the children are stolen at the end of the story. With these methods, we seek to answer my central question: can machine learning models learn and accurately predict the outcome of a story?

## Methods and Data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The data that was assembled was mostly compiled from a website that is hosted by a Professor at the University of Pittsburgh. Dr. D. L. Ashliman conducts folklore research and his website for Pied Piper stories are found [here](https://sites.pitt.edu/~dash/hameln.html). Only two stories were found outside of this source. Robert Browning's version was found from Project Gutenberg which is an online library of freely accessible books. The other was Viktor Dyk's story which is provided freely by Plamen Press. As we mentioned earlier, there are 22 different versions of the story. Each version of the story is stored inside of a `.txt` file. There are 10 different nationalities represented. Below is a table that displays the number works for each nationality and how many of those have the children stolen.

<center>

<b> Table 1 </b>

| Nationality|# of works|# of children stolen|# of children NOT stolen|
|--------|----|---|-----|
| German | 8 | 4  | 4 |
| UK | 5 | 5 | 0 |
| Czech | 1 | 1  | 0 |
| Austria | 2 | 0 | 2 |
| Italy | 1 | 0 | 1 |
| Denmark | 1 | 0 | 1 |
| Ireland | 1 | 0 | 1 |
| Iceland | 1 | 0 | 1 |
| France | 1 | 0 | 1 |
| Syria | 1 | 0 | 1 |

</center>


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As mentioned earlier, we use 3 models to predict the outcome of the stories. We used two different set of predictors to investigate which can be more indicative of revealing the outcome of the story. The first predictor, we tried to use was a sentiment score. The sentiment analysis tool we used was `spacytextblob` sentiment analysis tool. This is found in the `spaCy` toolkit library. It provides a sentiment score scaled between the range of -1 and 1. The second predictor we use was the TFIDF transformer provided from the `scikit-learn` toolkit. The 3 models that we implemented are:
1. K-Means Clustering
2. Logistic Regression
3. Support Vector Machine


#### Data Processing

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In order to get the most accurate sentiment score for an entire document, we chunked the entire txt file into a list of sentences. For each story, we used the `spacytextblob` sentiment tool to calculate a score between -1 and 1, where -1 is the most negative sentiment and 1 is the most positive sentiment. Since we want a single score for a document, we take the average of the sentence sentiment scores. However, before we do so, we take the absolute value of each item. It could be likely that writers with strong positive emotion could also have strong negative emotion in their writing in the same. If this is the case, then their average would be 0 which would indicate little emotion in the writing. Taking the absolute value of each score will eliminate this problem. However, our sentiment score, which shows the general emotion of writing, becomes more of a passion score. A passion score indicates if the writer outputs any emotion at all into the writing.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In order to achieve `TFIDF` matrix, we tokenized every document in which every word is a token. For each token, we made it lower-case and we also standardized the spelling of Hamelin. Some authors spelled the town either as "Hamelen" or "Hameln" which we changed to "Hamelin". Additionally, we eliminated stopwords from the list of tokens because they would add unnecessary noise to our model. Once there is a list of tokens for each document, we transformed the list into a `Counter` object. A `Counter` is a frequency dictionary where the `key` is the word, and the `value` is the number of times it appears in the document. Then the `Counter` objects are passed through the `scikit-learn` feature extraction method `DictVectorizer`. This method transforms dictionary-like objects (i.e. `Counter`) into vectors, `numpy` arrays, that can be used for later `scikit-learn` methods. Then using `scikit-learn`'s `TfidfTransformer` function, each vector is transformed into a normalized `TFIDF` representation. The goal of this transformation is to extract more information than what raw frequencies can provide. The goal of `TFIDF` is to scale the impact of frequently occurring words because they are less informative as they appear in almost every document of our dataset.

#### Machine Learning Methods

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The first method that we used was the unsupervised method K-Means Clustering Algorithm. The K-Means algorithm is a distance based algorithm that attempts to cluster samples into `K` number of groups. The algorithm creates `K` number of cluster points that are randomly assigned values. Once the points are assigned to the closest cluster, the cluster then moves to the average of the points, and this process repeats until the clusters converge on a point. In order to perform this analysis, we use the `scikit-learn` function `KMeans(n_clusters)`, where `n_clusters` is the number of clusters to initialize. The K-Means algorithm was run twice, once where the predictors were the passion scores and once where the predictors were the `TFIDF` representations.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The second method was the logistic regression. Logistic regression is a supervised learning Machine Learning technique, and is typically used to classify a categorical binary variable (1 or 0). The logistic regression is a statistical model which finds a linear combination of parameters in order to predict the log-odds of an event taking place. Similar to a linear regression, the logistic regression estimates the parameters (independent variables) in order to predict the probability of an event occurring. The logistic regression function we used is provided by the `scikit-learn` toolkit, in the `linear_model` library. The particular function is the `LogisticRegression()` function.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The third and final method is the Support Vector Machine (SVM). Support Vector Machines are a supervised classifying technique which often use different dimensionalities to find linear separated data. SVM does this by choosing a kernel which is a function to transform data into a higher dimensionality. Once there is a linear separable decision boundary between the classes, we need to find the support vectors. Support vectors are the closest points to the other class. Once support vectors are identified, the decision boundary is created by maximizing the distance between the different class' support vectors. It does this maximization in order to allow for test points with more variance to be classified correctly. SVMs can be run by the `scikit-learn` toolkit, `svm` library, and `linearSVC()` function.

## Results

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this section, we will discuss many analyses that we performed. First, we will discuss the overall results and trends in the sentiment analysis scores over time. Then we will discuss how each model performed when tasked to classify if the children were stolen or not. It will be discussed in the following order: K-Means Clustering, Logistic Regression, and Support Vector Machines. For each model, we will discuss the results and the reasoning behind making certain design decisions.

#### Sentiment Analysis

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As we explained earlier, the sentiment score is really a passion score where it measures the overall emotion of the author, not the actual positive or negative sentiment that a writer may exhibit. The results of the sentiment score do not appear to be amazing unfortunately. There does not appear to be any clear pattern of overall trends of the stories nor does there appear to be any patterns within groups of the outcomes. Below in Figure 1, we can observe that the earlier versions of the story tended to have the children abducted, and the stories written past 1750 are more likely to not mention the children or to replace the children with livestock. Additionally, the graph does not indicate that one class was more likely to have more passion in the writing. This is verified by a `t-test` of the two class' sentiment scores. The `p-value` of the score was approximately .38% and the `t-score` was -.904, both very poor scores. This furthers our findings that the sentiment scores are very similar and may not provide any discernable insights. The test was performed using the `scipy` library and the `ttest_ind(group1, group2)` function.

<center> <b> Figure 1 </b>

![image1](./Figs/Sentiment_graph3.png)
</center>

#### K-Means Clustering

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The first K-Means analysis that we completed was using the sentiment scores as the predictors. Since the sentiment scores are 1 dimensional, the clusters are essentially being created on a single line instead of a graph. This makes the K-Means algorithm extremely simple since it is only trying to cluster on one axis. The goal of our implementation of the K-Means algorithm is to have one class be clustered together, and the other class being classified with another cluster. Below in Table 2 are the results of the K-Means algorithm with sentiment analysis as the predictor. For review, `precision` is the number of positive classes that actually belong to the positive class. `Recall` measures the number of positive class predictions made out of all the positive data points in the dataset. The `f1-score` is a "harmonic" mean of precision and recall. Since it is impossible to maximize both, we can maximize the f1-score to find a statistic to optimize. Finally, `support` referred to the number of samples. As we can see in Table 2, our precision, recall, and f1-score are not very good at all. The `False` class (children not stolen) scores only at 50% meanwhile the `True` class (children stolen) scores around %40. The overall accuracy is .45% which is worse than randomly choosing a prediction. It is clear that is not a great model to predict the outcome of the story.


<center>

<b> Table 2: K-Means (sentiment scores) </b>

| | Precision | Recall | f1-score | Support|
|--|:---:|:---:|:---:|:--:|
| False | .50 | .50 | .50 | 12 |
| True | .40 | .40 | .40 | 10 |
| Accuracy | - | - | .45 | 22 |

</center>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The second K-Means model we created was with the `TFIDF` representation as the independent variables. K-Means finding the closest cluster based on distance metrics is still true with a `TFIDF` representation, but it is very abstract because it is finding the distances of every point to a cluster in 2500 dimensions. Below is Table 3 where we can examine how well the model performed. We can see the results were drastically more successful. The overall accuracy was .82 which indicates that 82% of the samples were classified to the correct cluster. For precision, the `False` category scored .79, and the `True` category .88 which indicates that their respective class was able to predict that class 79% and 88% of the time. The `False` category scored .92 for recall which claims that 92% of the `False` documents were classified into the same cluster. The `True` category only scored .78. The f1-scores for `False` and `True` were .85 and .78, respectively. This model provided more accurate results. It appears that the model struggled with classifying the `True` category more so . This could be because of small sample sizes, and that the `False` technically has 20% more documents in its respective class.

<center>

<b> Table 3: K-Means (`TFIDF` representation) </b>

| | Precision | Recall | f1-score | Support|
|--|:---:|:---:|:---:|:--:|
| False | .79 | .92 | .85 | 12 |
| True | .88 | .70 | .78 | 10 |
| Accuracy | - | - | .82 | 22 |

</center>

#### Logistic Regression

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Due to the poor results of the results of the sentiment analysis, we decided to only analyze the `TFIDF` representation for the logistic regression as well as for the support vector machine. The logistic regression scored some interesting results. Again, our total sample is very small, but we still wanted to test our model so our training sample is even smaller than what we used in the K-Means models. The logistic model returned some very interesting results. Below is Table 4 where we will evaluate the performance statistics of the model. We can see that the precision for both `False` and `True` categories scored well, .67 and 1.00. While .67 maybe appear low, only one document that was classified as `False` that should have been true meanwhile all the predicted `True` categories were predicted correctly. Conversely, it was the `False` category that excelled in the recall statistic meanwhile the `True` category struggled. This suggests all the `False` documents were identified as such as meanwhile the `True` category struggled in the overall classification process (33%). Finally, we see through the f1-score that the `False` category (.80) was a much more successful than the `True` category (.50). The overall accuracy of the model was 71%.

<center>

<b> Table 4: Logistic Regression </b>

| | Precision | Recall | f1-score | Support|
|--|:---:|:---:|:---:|:--:|
| False | .67 | 1.00 | .80 | 4 |
| True | 1.00 | .33 | .50 | 3 |
| Accuracy | - | - | .71 | 7 |

</center>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Because we are using a type of linear regression model where we are optimizing the parameters, there is great insight into checking which parameters have the most impactful weights. In this case, the parameters are actually the words so we can analyze which words have the highest impact on the model. Below is a Table 5, a table of the words with the most impactful coefficients. On the left side are the words and coefficients that increase the probability of the children being stolen and on the right are the words and coefficients that decrease the probability of the children being stolen. We can see that the words `children`, `piper`, and `ratcatcher` increase the probability of the children the most. Conversely, the highest negative coefficients are `mice`,`mill` and `king`. Some words in each list are not surprising such as `children` being an indicator of the children being stolen or `drive` as a word that indicates the children were not stolen ("drive the livestock"). However, there are some words such as `piper`, `rats`, and `pipe` that could be indicative of any classification.

<center>

<b> Table 5: Logistic Regression Coefficients </b>

| positive Words | Coefficients || Negative Words | Coefficients |
|--|:---:||:---:|:--:|
| children | 0.268061 || mice | -0.235544 |
| piper | 0.266879 || mill | -0.202106 |
| ratcatcher | 0.214702 || king | -0.193398 |
| town | 0.205866 || rats | -0.165204 |
| street | 0.192490 || ummanz | -0.160722 |
| pipe | 0.160415 || vermin | -0.160181 |
| fife | 0.159172 || drive | -0.149433 |
| hamelin | 0.150889 || presented | -0.139831 |
| plague | 0.140826 || forth | -0.135921 |
| good | 0.133055 || soil | -0.128578 |

</center>


#### Support Vector Machine

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The support vector machine scored very well when using the `TFIDF` represention of the documents. Below is Table 6 where we analyze the precision, recall and f1-score of the model. The `False` category scored a 1.00 for precision, .75 for recall and .86 for f1-score. The `True` category scored a .75 for precision, 1.00 for recall and .86 for f1-score. As we can see this support vector machine model performed very well as its overall accuracy is 86%, the highest accuracy that a model has performed. Additionally, in Table 7 we can analyze the words with the highest coefficients. The list are almost identical with the logistic regression for both the positive words and their coefficients and the negative words and their coefficients. The main difference between the two models are the scales of the coefficients of the parameters. The weights of the SVM models are much greater in both the positive and negative word lists.

<center>

<b> Table 6: Support Vector Machine </b>

| | Precision | Recall | f1-score | Support|
|--|:---:|:---:|:---:|:--:|
| False | 1.00 | .75 | .86 | 4 |
| True | .75 | .100 | .86 | 3 |
| Accuracy | - | - | .86 | 7 |

</center>

<center>

<b> Table 7: Support Vector Machine Coefficients </b>

| positive Words | Coefficients || Negative Words | Coefficients |
|--|:---:||:---:|:--:|
| children | 0.421570 || mice | -0.388993 |
| piper | 0.404780 || mill | -0.363940 |
| ratcatcher | 0.346224 || king | -0.322533 |
| town | 0.318906 || rats | -0.279559 |
| street | 0.291060 || vermin | -0.255829 |
| fife | 0.261146 || ummanz | -0.253778 |
| pipe | 0.246512 || drive | -0.231776 |
| hamelin | 0.238612 || presented | -0.225598 |
| plague | 0.224091 || forth | -0.208997 |
| good | 0.207727 || magician | -0.208120 |

</center>


## Analysis & Discussion

#### Sentiment Analysis

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The sentiment analysis did not perform as well as we thought it would. Our hypothesis was that authors who wrote children that were stolen would score higher in the sentiment/passion scores. However, as we showed in the general trends of the sentiment analysis scores, there was no real clear pattern in the data. Moreover, it was a terrible predictor when applied to the K-Means algorithm. I do think that there could be potential use of sentiment analysis in the future for algorithms learning outcomes of stories. There are many different ways to measure sentiment and many tools that use different methods to do so. Therefore, we feel strongly that there is good reason to investigate further into sentiment analysis as a method to learn outcomes.

#### Machine Learning methods

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; As we discussed earlier, the K-Means algorithm was actually the second most successful model that we had. It is clear from this analysis that a distance based algorithm could be very successful. Further research could investigate the performance of the `K-Nearest Neighbors` algorithm, the supervised learning version of K-Means. Because of the small amount of data, an unsupervised method was helpful because we could use all of the data as the 'training' set for K-Means. Meanwhile for supervised methods, we have to split up our dataset into training and test sets which incur more bias into our training set. For this reason, the K-Means was a successful method providing 82% accuracy on the test set.

 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The logistic model did not perform as well as the K-Means model and was actually the worst performing model of the three models. Its overall accuracy was 71%. It is unclear why the model was not as successful, but there are some interesting insights to be studied in the words' coefficients. One word that is particularly interesting is `Hamelin` because this would imply that all German versions have the children stolen as the outcome. Perhaps nationality could have been a good way to predict if the children were stolen. We refrained from doing this analysis because our goal is to find patterns that could go beyond the writer's demographics. This is because we want to find characteristics of the work itself that could be indicative of the outcome of stories. Overall, the logistic regression did not provide a terrible model, but we have already seen others that were more accurate.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The Linear Support Vector Machine was overall the most accurate and informative model that we were able to create. Its overall accuracy was 86%, a very impressive mark considering the accuracy of the other models. Computationally, it is unclear why the Support Vector Machine would perform so much better. However, we can still understand the results relatively well. The interesting fact about the SVM model's coefficients was that the top 10 coefficients were essentially the same as the logistic regression. The main difference was that the scale of the coefficients was so much greater in the SVM. The SVM model gave more influence to certain words than the logistic regression would and as a result, it performed 15 percentage points better than the logistic regression did.

### Conclusion

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Overall, this was a tough problem to tackle because it is hard to conceptualize how we can get a machine to learn outcomes of stories. In essence, we were trying to model the context and story arcs of the same story. I think we succeeded, but we also had some major limitations. The sample size is a big problem regarding this project. We were only able to identify 22 versions of `The Pied Piper`. I think finding more versions could make our models stronger and our interpretations more valid. However, I think this project could be useful in the domain knowledge of natural language processing as the field is on its way to teach computers to conceptualize meaning in stories.

## References

Dirckx, J. H. (1980). The pied Piper of Hamelin. A medical-historical interpretation. The American Journal of Dermatopathology, 2(1), 39–45. https://doi.org/10.1097/00000372-198000210-00007

Pied Piper of Hameln. (n.d.). Retrieved April 24, 2022, from https://sites.pitt.edu/~dash/hameln.html

Sklearn.svm.LinearSVR. (n.d.). Scikit-Learn. Retrieved April 24, 2022, from https://scikit-learn/stable/modules/generated/sklearn.svm.LinearSVR.html
