# Parallel depiction of societal topics in movies and news : To what extent does cinema capture the key societal issues of its time, and how is this in sync with their representation in newspapers?


## Abstract 
Movies often reflect the culture and society of a certain time and represent topics people are interested in. Our project explores how the themes and topics of movies have evolved over time, correlating them with trends observed in newspapers. Using the CMU Movie Summary Corpus, which includes data from over 42000 movies, our goal is to determine how societal issues are present in movies and how this presence evolves in comparison to the same topics in news articles. We want to examine their potential correlations and how they may appear shifted in time. To achieve this goal, we aim to detect important topics via topic modeling in both the CMU dataset and a news article dataset from the New York Times. From that, we will be able to compare the representation of these topics over time.

## Data Story



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
- Metadata: keep only the year of release of the movies and categorize the movies by decade to better visualize the evolution across time, clean the genres of the movies by regrouping redundant ones.

2. **Numerical representation for words:**: 
We used the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model from Hugginf Face, with the python library "Sentence Transformers" (SBERT). This is a pre-trained model that allows us to easily transform a word or a bag of words to a high-dimensional vector representation (embedding).


3. **Identify societal themes**:
The noisy caracter of the movie descriptions made it difficult to extract reccurent topics. We thus used the News dataset to identify common topics with Latent Dirichelet Alocation, and then choose a few themes to analyse in detail manually. 
4. **Queries by themes**:
With the identified societal themes, we computed the vector embedding of a manually chosen querry representing each theme. We were then able to compute the cosine similarity between each movie summary or news article's embedding and the querry's embedding. 
5. **Analyze relationship between themes in both movies and news articles**:
We analyzed the relative presence of themes in both datasets over time, focusing on the cross-correlation between the two distributions. We look for any interesting statistical correlation. We provide plots to visualize these trends and provide possible explanation of the trends identified.
6. **In-depth analysis: societal reflections in movies via genre and sentiment**:
We explore how those societal topics are presented in movies and how they reflect society at various points in time. Typically, we look for genres in which they are present and the kind of sentiments they are associated with -- positive emotions, negative, fear, hope, etc, in order to see what kind of depiction of society is preferred in movies.
7. **Conclusion and Data Story**:
We created a nice interface to present our findings, display plots and draw conclusions from what was observed in the previous steps.

## Structure of the repository
```plaintext
ADA-2024-PROJECT-ASTONISHINGDATAAVENGERS24/
│
├── data/
│   ├── df_movies/ #movies dataset per theme
│   ├── df_news/ # news dataset per theme
│
├── src/
│   ├── preprocess/
│   ├── sentiment_analysis/ # contains all notebooks per theme
│   ├── themes/ # notebook for finding themes in movies and news
│   ├── utils/ #helpers 
│
├── README.md
├── results.ipynb
```


## Proposed Timeline  

| **Date**          | **Task**                                                                                     |
|--------------------|---------------------------------------------------------------------------------------------|
| 28.10.24 – 15.11.24 | Find topic, data preprocessing / cleaning, initial analyses on plot summaries, news topics and movie genres. |
| 18.11.24 – 24.11.24 | Homework 2   |
| 25.11.24 – 01.12.24 | Finish movies and news topic modeling, start in-depth analysis     |
| 02.12.24 – 08.12.24 | Visualizations, draft preliminary conclusions about the correlation between movie and newspaper topics |
| 09.12.24 – 15.12.24 | Report draft                                         |
| 16.12.24 – 22.12.24 | Finalize the project: website, code cleaning, GitHub organization, and final presentation.  |

---

## Organization Within the Team  

| **Task**                     | **Team Member(s)**   |
|-------------------------------|----------------------|
| Data Cleaning and Processing  | Everyone |
| Movies topic modeling | Antoine / Massimo |
| News topic modeling               | Clémentine |
| Topics queries and in-depth analysis   | Everyone, 1 topic per group member |
| Genre and Sentiment Analysis           | Sarah / Shuli |
| Report Writing                | Everyone |

---
## Contributions
|  **Team Member**                    |  **Task**  |
|-------------------------------|----------------------|
| Sarah  | plots and algorithms for genre and sentiment analysis, analysis of themes gender equality and cold war|
| Massimo | Data story, analysis of themes WWII|
| Antoine  | Extract news from themes, correlation between movies and themes|
| Clémentine | Theme findings, extracting movies from themes, analysis of theme technology and vietnam war|
| Shuli          | plots and algorithms for genre and sentiment analysis, analysis of themes health, Data Story |
