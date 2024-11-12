# Topics of Interest in Society Through Movies

## Abstract [To Be Completed]
Movies often reflect the culture and society of a certain time and represent topics people are interested in. Our project explores how the themes and topics of movies have evolved over time, correlating them with trends observed in newspapers. Using the CMU Movie Summary Corpus, which includes data of over 42000 movies, our goal is to determine whether a topic is first discussed in media before being presented in movies or if it is the opposite.

---

## Research Questions  

    1. Evolution of topics in movies over time: how have specific themes in movies evolved over time? Do we see certain themes gaining popularity in movies after some events happening in society?
    2. Do movies tend to lead discussions about certain topics in society (e.g. environmental issues) or do they reflect themes that are already popular in other media?
    3. When movies talk about topics that are trending in society, how do their approaches vary depending on the nature of the topic? What genres do they use to portray these topics?
    4. If a topic is recurrent in the news, do the movies in this period of time tend to represent the topic, or try to provide an escape to the audience by producing more movies that can get their attention away from the news? 
 

---

## Proposed Additional Datasets  
- **New York Times Articles Dataset**: (~2GB)

---

## Methods  

### Data Preparation  
1. **Data Preprocessing**:  
   - **Plot Summaries**: Remove stop words and punctuation to keep only keywords.  
   - **Metadata**:  
     - Extract the year of release and group movies by decade to better visualize evolution of genres over time.  
     - Clean and regroup genres to avoid redundancy (e.g., regrouping 'Anti-war' and 'Anti-war film').  

2. **Genre Trends**:  
   - Analyze the distribution of movie genres across decades and continents using metadata.
   - Compute first the top 5 genres in the whole world, and how these genres are distributed in each continent, how popular they are in each continent.

---

### Models  
- Use word embedding techniques to transform text into vector representations.  
- Cluster movies based on thematic similarity using vector representations.

---

### Analysis  
1. **Plot Summaries**:  
   - Identify prevalent topics and themes for each decade [to be completed].  

2. **Merging with News Data**:  
   - Analyze the correlation between trends in movie themes and societal trends observed in news articles [to be completed].  

---

## Proposed Timeline  

| **Date**          | **Task**                                                                                     |
|--------------------|---------------------------------------------------------------------------------------------|
| 28.10.24 – 15.11.24 | Set up a topic, start data preprocessing/cleaning, and perform initial analyses on plot summaries and news topics. |
| 18.11.24 – 24.11.24 | Homework 2                                                                                 |
| 25.11.24 – 01.12.24 | Finalize algorithms for analyzing plot summaries and identify relevant news articles.      |
| 02.12.24 – 08.12.24 | Create visualizations, draft preliminary conclusions about the correlation between movie and newspaper topics. |
| 09.12.24 – 15.12.24 | Begin report drafting to present the data story.                                           |
| 16.12.24 – 22.12.24 | Finalize the project: website, code cleaning, GitHub organization, and final presentation.  |

---

## Organization Within the Team  

| **Task**                     | **Team Member(s)**   |
|-------------------------------|----------------------|
| Data Cleaning and Processing  | [Names to be filled] |
| Implementation and First Analyses | [Names to be filled] |
| Final Analyses                | [Names to be filled] |
| Visualization                 | [Names to be filled] |
| Report Writing                | [Names to be filled] |

---
