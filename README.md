# Parallel depiction of societal topics in movies and news : How does cinema capture the key societal issues of its time, and how is this in sync with their representation in newspapers?


## Abstract 
Movies often reflect the culture and society of a certain time and represent topics people are interested in. Our project explores how the themes and topics of movies have evolved over time, correlating them with trends observed in newspapers. Using the CMU Movie Summary Corpus, which includes data from over 42000 movies, our goal is to determine how societal issues are present in movies and how this presence evolves in comparison to the same topics in news articles. We want to examine their potential correlations and how they may appear shifted in time. To achieve this goal, we aim to detect important topics via topic modeling in both the CMU dataset and a news article dataset from the New York Times. From that, we will be able to compare the representation of these topics over time.

## Data Story:
https://mass-14.github.io/2024-12-18-inspector-gADAget/



---

## Research Questions  
1. How have specific themes in movies evolved over time? Do we see certain themes gaining popularity in movies after some events happening in society?

2. Do movies initiate discussions about certain topics (e.g. environmental issues) or do they reflect themes that are already popular in newspapers?

3. When movies address popular topics, how do their storytelling approaches differ based on the type of topic? Which genres are commonly chosen to represent these themes?

4. When a topic appear over a certain period, do movies from that time tend to address and represent the issue, or do they provide an escape by focusing on unrelated themes? Additionally, how do the feelings portrayed in these films align with public sentiment?


---

## Proposed Additional Datasets  
- **New York Times Articles Dataset**: (~2.93GB)
  This dataset comprises titles and excerpts from New York Times articles from 1920 to 2020, found at https://www.kaggle.com/datasets/tumanovalexander/nyt-articles-data. It will be used to find recurrent topics, analyze the presence of certain themes over time, and compare the results to the movies dataset
 

---

## Methods  

### Steps 
1. **Data Preprocessing**:
We removed stop words and punctuation, to keep only keywords in the movies plot summaries and NYT articles.
Metadata: we kept only the year of release of the movies and categorize the movies by decade to better visualize the evolution across time, cleaned the genres of the movies by regrouping redundant ones.

2. **Identify societal themes**:
The noisy caracter of the movie descriptions made it difficult to extract reccurent topics. We thus used the News dataset to identify common topics with Latent Dirichelet Allocation (LDA) or Latent Semantic Indexing (LSI) to find some inspiration, and also chose manually a few themes to analyse in detail .

3. **Numerical representation for words**:
We used the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model from Hugging Face, with the python library "Sentence Transformers" (SBERT). This is a pre-trained model that allows us to easily transform a word or a bag of words to a high-dimensional vector representation (embedding) and perform semantic search.

4. **Queries by themes**:
With the identified societal themes, we computed the vector embeddings of a manually chosen query representing each theme. We then computed the cosine similarity between each movie summary or news article embedding and the query's embedding.

5. **Analyze relationship between themes in both movies and news articles**:
We analyzed the relative presence of themes in both datasets over time, focusing on the cross-correlation between the two distributions. We looked for any interesting statistical correlations. We provided plots to visualize these trends and offer possible explanations for the trends identified.

6. **In-depth analysis: societal reflections in movies via genre and sentiment**:
We explored how these societal topics are presented in movies and how they reflect society at various points in time. Typically, we looked at the genres in which they are present and the kinds of sentiments they are associated with — positive emotions, negative emotions, fear, hope, etc., to understand what kind of depiction of society is preferred in movies.

7. **Conclusion and Data Story**:
We created a nice interface to present our findings, display plots and draw conclusions from what was observed in the previous steps.

## Structure of the repository
```plaintext
ADA-2024-PROJECT-ASTONISHINGDATAAVENGERS24/
│
├── data/
│   ├── df_movies/ # movies dataset per theme
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

---
## Contributions
|  **Team Member**                    |  **Task**  |
|-------------------------------|----------------------|
| Sarah  | Plots and algorithms for genre and sentiment analysis, analysis of gender equality and Cold War themes|
| Massimo | Data story, analysis of WWII theme|
| Antoine  | Extracting news by themes, analysis of the correlation between movies and themes|
| Clémentine | Extracting movies by themes, analysis of Technology and Vietnam War themes|
| Shuli          | Plots and algorithms for genre and sentiment analysis, analysis of Health theme, Data Story |
