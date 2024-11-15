# Parallel depiction of societal topics in movies and news : To what extent does cinema capture the key societal issues of its time, and how is this in sync with their representation in newspapers?


## Abstract 
Movies often reflect the culture and society of a certain time and represent topics people are interested in. Our project explores how the themes and topics of movies have evolved over time, correlating them with trends observed in newspapers. Using the CMU Movie Summary Corpus, which includes data from over 42000 movies, our goal is to determine how societal issues are present in movies and how this presence evolves in comparison to the same topics in news articles. We want to examine their potential correlations and how they may appear shifted in time. To achieve this goal, we aim to detect important topics via topic modeling in both the CMU dataset and a news article dataset from the New York Times. From that, we will be able to compare the representation of these topics over time.


---

## Research Questions  
1. How have specific themes in movies evolved over time? Do we see certain themes gaining popularity in movies after some events happening in society?
2. Do movies initiate discussions about certain topics (e.g. environmental issues) or do they reflect themes that are already popular in newspapers?
3. When movies address popular topics, how do their storytelling approaches differ based on the type of topic? Which genres are commonly chosen to represent these themes?
4. When a topic frequently appears in the news, do movies from that time tend to address and represent the issue, or do they provide an escape by focusing on unrelated themes? Additionally, how do the feelings portrayed in these films align with public sentiment?


---

## Proposed Additional Datasets  
- **New York Times Articles Dataset**: (~2.93GB)
  This dataset comprises title and excerpt for New York times articles from 1920 to 2020 found on https://www.kaggle.com/datasets/tumanovalexander/nyt-articles-data. It will be used to find recurrent topics over periods of time via topic modeling.
- **English Words**
  https://github.com/dwyl/english-words?tab=readme-ov-file to filter movie descriptions, in particular the characters and actor’s names that were introducing significant noise. 

  

---

## Methods  

### Steps 
1. **Data Preprocessing**:  
    - Remove stop words and punctuation, to keep only keywords in the movies plot summaries and NYT articles.
    - Metadata: keep only the year of release of the movies and categorize the movies by decade to better visualize the evolution across time, clean the genres of the movies as there are many redundant genres that can be combined in one. 
    - Word2Vec model word fitting:
          - Find a proper representation for words not in the model, or remove them.
          - Remove words that have a representation but are not in the English language (mainly names). 
2. **Numerical representation for words, Word2Vec model from Gensim**: This NLP technique usually uses a large neural network to create word embeddings: a map from words to a vector representation in a high-dimensional space. The Word2Vec model is usually trained on a very large corpus (in our case, we use a model trained on a dump of the entire English Wikipedia) and  maps vectors with similar usage patterns to similar vectors. 
3. **Numerical representation for movie descriptions**: **Term Frequency-Inverse Document Frequency** is a method to compute a numerical representation for a movie description. A common pitfall when combining individual vectors of words to represent a text is that all  words should not have the same influence. To solve this, TF-IDF computes a weight for each word of the sentence, that will determine its global impact. If a word appears often in other movie descriptions, it will have a lower score. This follows the idea that rare words are generally more precise and meaningful than common ones. 
4. **Identify societal themes**:
    We take the news dataset we already preprocessed and we find the top 30 topics per decade by topic modeling. We’ll first try different topic modeling techniques, such as Latent Dirichlet Analysis (LDA) to find what seems the most suitable. We’ll find words to characterize such themes by finding an average word vector from the theme words and look for the closest word in our dictionary. We’ll select manually the most meaningful themes for further analysis and semantic search. 
5. **Queries by themes**:
   Now that we have the societal themes we were looking for, we perform semantic search queries on both news and movies datasets to find the most relevant articles and plots to the theme. If the topics retrieved from the previous steps are not sufficient, we will complement our analysis by doing some queries with other themes we find interesting.
6. **Analyze relationship between themes in movies and in news articles**
    We analyze the relative presence of themes for both datasets over time, visualize these trends and look for any interesting statistical correlation
7. **In-depth analysis: societal reflections in movies via genre and sentiment**
    We also want to explore how those societal topics are presented in movies and how they reflect society at various points in time. Typically, we’ll look for genres in which they are present, the kind of sentiments they are associated with, i.e. positive emotions, negative, fear, hope, etc. We’ll analyze all that to see what kind of depiction of society is preferred in movies by a combination of all those elements.
8. **Conclusion and Data Story**
We’ll create a nice interface to present our findings, display our plots and draw conclusions from what we observed in the previous steps.
---

## Proposed Timeline  

| **Date**          | **Task**                                                                                     |
|--------------------|---------------------------------------------------------------------------------------------|
| 28.10.24 – 15.11.24 | Find topic,data preprocessing/cleaning, initial analyses on plot summaries, news topics and movie genres. |
| 18.11.24 – 24.11.24 | Homework 2   |
| 25.11.24 – 01.12.24 | Finish movies topic modeling and news topic modeling, start in-depth analysis     |
| 02.12.24 – 08.12.24 | Visualizations, draft preliminary conclusions about the correlation between movie and newspaper topics |
| 09.12.24 – 15.12.24 | Report draft                                         |
| 16.12.24 – 22.12.24 | Finalize the project: website, code cleaning, GitHub organization, and final presentation.  |

---

## Organization Within the Team  

| **Task**                     | **Team Member(s)**   |
|-------------------------------|----------------------|
| Data Cleaning and Processing  | Everyone |
| Movies topic modeling | Antoine/Massimo|
| News topic modeling               | Clémentine |
| Topics queries and in-depth analysis   | 1 topic per group member (Everyone) |
| Genre and Sentiment Analysis           | Sarah / Shuli|
| Report Writing                | Everyone |

---
