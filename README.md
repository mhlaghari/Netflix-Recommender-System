# Netflix Recommendation System 
In this project, I demonstrate 2 ways of building a recommender system:
1. Popularity Based
2. Content Based 

### Popularity based
A popularity based recommender system provides the most popular searches, 

` For example, Top 10 movies watched in your region/ country`

### Content based 
A Content based recommender system provides the most relevant searches in regards to the content being consumed 

` For Example: Watching The Godfather will give you the top N movies similar to The Godfather`


```python
import pandas as pd 
import numpy as np 
```


```python
movie_names = pd.read_csv('movies (1).csv')
movie_names.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings = pd.read_csv('ratings.csv')
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
  </tbody>
</table>
</div>




```python
# df = pd.merge(movie_names, ratings, how='left', 
#               on='movieId')
```


```python
# df.shape
```


```python
df2 = pd.merge(movie_names, ratings, on='movieId')
df2.shape
```




    (100004, 6)



## **Criteria** for Popularity Based Recommendation System

The Criteria is based on:
1. Movies with the highest rating
2. Number of views


```python
df2.groupby('title')['rating'].mean().sort_values(ascending=False).head()
```




    title
    Ivan Vasilievich: Back to the Future (Ivan Vasilievich menyaet professiyu) (1973)    5.0
    Alien Escape (1995)                                                                  5.0
    Boiling Point (1993)                                                                 5.0
    Bone Tomahawk (2015)                                                                 5.0
    Borgman (2013)                                                                       5.0
    Name: rating, dtype: float64




```python
df2.groupby('title')['rating'].count().sort_values(ascending=False).head()
```




    title
    Forrest Gump (1994)                          341
    Pulp Fiction (1994)                          324
    Shawshank Redemption, The (1994)             311
    Silence of the Lambs, The (1991)             304
    Star Wars: Episode IV - A New Hope (1977)    291
    Name: rating, dtype: int64




```python
ratings_mean_count = pd.DataFrame(df2.groupby('title')['rating'].mean())
ratings_mean_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>"Great Performances" Cats (1998)</th>
      <td>1.750000</td>
    </tr>
    <tr>
      <th>$9.99 (2008)</th>
      <td>3.833333</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>'Neath the Arizona Skies (1934)</th>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>2.250000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>xXx (2002)</th>
      <td>2.478261</td>
    </tr>
    <tr>
      <th>xXx: State of the Union (2005)</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>¡Three Amigos! (1986)</th>
      <td>3.258065</td>
    </tr>
    <tr>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
      <td>4.500000</td>
    </tr>
    <tr>
      <th>İtirazım Var (2014)</th>
      <td>3.500000</td>
    </tr>
  </tbody>
</table>
<p>9064 rows × 1 columns</p>
</div>




```python
ratings_mean_count['rating_counts'] = pd.DataFrame(df2.groupby('title')['rating'].count())
ratings_mean_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>rating_counts</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>"Great Performances" Cats (1998)</th>
      <td>1.750000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>$9.99 (2008)</th>
      <td>3.833333</td>
      <td>3</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>2.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Neath the Arizona Skies (1934)</th>
      <td>0.500000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>2.250000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>xXx (2002)</th>
      <td>2.478261</td>
      <td>23</td>
    </tr>
    <tr>
      <th>xXx: State of the Union (2005)</th>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>¡Three Amigos! (1986)</th>
      <td>3.258065</td>
      <td>31</td>
    </tr>
    <tr>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
      <td>4.500000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>İtirazım Var (2014)</th>
      <td>3.500000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>9064 rows × 2 columns</p>
</div>




```python
ratings_mean_count['rating'] = round(ratings_mean_count['rating'],1)
```


```python
ratings_mean_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>rating_counts</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>"Great Performances" Cats (1998)</th>
      <td>1.8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>$9.99 (2008)</th>
      <td>3.8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Neath the Arizona Skies (1934)</th>
      <td>0.5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>2.2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>xXx (2002)</th>
      <td>2.5</td>
      <td>23</td>
    </tr>
    <tr>
      <th>xXx: State of the Union (2005)</th>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>¡Three Amigos! (1986)</th>
      <td>3.3</td>
      <td>31</td>
    </tr>
    <tr>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
      <td>4.5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>İtirazım Var (2014)</th>
      <td>3.5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>9064 rows × 2 columns</p>
</div>




```python
ratings_mean_count = ratings_mean_count[(ratings_mean_count['rating'] > 3) & (ratings_mean_count['rating_counts'] > 100)]
```


```python
ratings_mean_count = ratings_mean_count.sort_values(by='rating', ascending=False)
ratings_mean_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>rating_counts</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Godfather, The (1972)</th>
      <td>4.5</td>
      <td>200</td>
    </tr>
    <tr>
      <th>Shawshank Redemption, The (1994)</th>
      <td>4.5</td>
      <td>311</td>
    </tr>
    <tr>
      <th>Usual Suspects, The (1995)</th>
      <td>4.4</td>
      <td>201</td>
    </tr>
    <tr>
      <th>Godfather: Part II, The (1974)</th>
      <td>4.4</td>
      <td>135</td>
    </tr>
    <tr>
      <th>Pulp Fiction (1994)</th>
      <td>4.3</td>
      <td>324</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>Cliffhanger (1993)</th>
      <td>3.1</td>
      <td>106</td>
    </tr>
    <tr>
      <th>Dumb &amp; Dumber (Dumb and Dumber) (1994)</th>
      <td>3.1</td>
      <td>158</td>
    </tr>
    <tr>
      <th>Home Alone (1990)</th>
      <td>3.1</td>
      <td>129</td>
    </tr>
    <tr>
      <th>Mask, The (1994)</th>
      <td>3.1</td>
      <td>157</td>
    </tr>
    <tr>
      <th>Net, The (1995)</th>
      <td>3.1</td>
      <td>102</td>
    </tr>
  </tbody>
</table>
<p>145 rows × 2 columns</p>
</div>



So lets suppose that you make a subset of movies being watched in the region, you can take the count of films being watch for that region

## Content Based Recommender System 

Calculating Cosine Similarity


```python
from math import *

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)* square_rooted(y)
    return round(numerator/float(denominator))
```


```python
from sklearn.metrics.pairwise import cosine_similarity # performs same work as the cosine similarity we created above
from sklearn.feature_extraction.text import CountVectorizer 

pd.set_option('display.max_columns', 100)
new_movies = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')
```


```python
df = new_movies[['Title', 'Genre', 'Director', 'Actors', 'Plot']]
```


```python
#Discarding the commas between actors' full names and getting only the first three names
df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])
```

    /var/folders/4y/3qvgxp5d62l2_0f4sx2hgqdr0000gn/T/ipykernel_90801/4222200124.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])



```python
# putting the genres in a list of words
df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))
```

    /var/folders/4y/3qvgxp5d62l2_0f4sx2hgqdr0000gn/T/ipykernel_90801/1136174772.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))



```python
df['Director'] = df['Director'].map(lambda x: x.split(' '))
```

    /var/folders/4y/3qvgxp5d62l2_0f4sx2hgqdr0000gn/T/ipykernel_90801/795600105.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['Director'] = df['Director'].map(lambda x: x.split(' '))



```python
#convert the actors names to lower case do avoid duplicates. Example, so that names like 'ROBBIN' and 'robbin' will not be repeated.

for index, row in df.iterrows():
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
    row['Director'] = ''.join(row['Director']).lower()
```


```python
import rake_nltk
from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/mhlaghari/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    [nltk_data] Downloading package punkt to /Users/mhlaghari/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.





    True




```python
df['Key_words'] = ''

for index, row in df.iterrows():
    plot = row['Plot']
    
    #instantiating Rake
    r = Rake()
    
    #extracting key workds by passing the text
    r.extract_keywords_from_text(plot)
    
    #Getting the dictionary with key workds and their scores 
    key_words_dict_scores = r.get_word_degrees()
    
    #assigning the key words to the new column
    row['Key_words'] = list(key_words_dict_scores.keys())
    
# Dropping the plot column
df.drop('Plot', axis=1, inplace=True)
    
```

    /var/folders/4y/3qvgxp5d62l2_0f4sx2hgqdr0000gn/T/ipykernel_90801/1961471345.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['Key_words'] = ''
    /var/folders/4y/3qvgxp5d62l2_0f4sx2hgqdr0000gn/T/ipykernel_90801/1961471345.py:19: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df.drop('Plot', axis=1, inplace=True)



```python
key_words_dict_scores
```




    defaultdict(<function rake_nltk.rake.Rake._build_word_co_occurance_graph.<locals>.<lambda>()>,
                {'mumbai': 3,
                 'teen': 3,
                 'reflects': 3,
                 'upbringing': 1,
                 'slums': 1,
                 'accused': 1,
                 'cheating': 1,
                 'indian': 2,
                 'version': 2,
                 'wants': 1,
                 'millionaire': 2,
                 '?"': 2})




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Genre</th>
      <th>Director</th>
      <th>Actors</th>
      <th>Key_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Shawshank Redemption</td>
      <td>[crime,  drama]</td>
      <td>frankdarabont</td>
      <td>[timrobbins, morganfreeman, bobgunton]</td>
      <td>[two, imprisoned, men, bond, number, years, fi...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Godfather</td>
      <td>[crime,  drama]</td>
      <td>francisfordcoppola</td>
      <td>[marlonbrando, alpacino, jamescaan]</td>
      <td>[aging, patriarch, organized, crime, dynasty, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Godfather: Part II</td>
      <td>[crime,  drama]</td>
      <td>francisfordcoppola</td>
      <td>[alpacino, robertduvall, dianekeaton]</td>
      <td>[early, life, career, vito, corleone, 1920s, n...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Dark Knight</td>
      <td>[action,  crime,  drama]</td>
      <td>christophernolan</td>
      <td>[christianbale, heathledger, aaroneckhart]</td>
      <td>[menace, known, joker, emerges, mysterious, pa...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12 Angry Men</td>
      <td>[crime,  drama]</td>
      <td>sidneylumet</td>
      <td>[martinbalsam, johnfiedler, leej.cobb]</td>
      <td>[jury, holdout, attempts, prevent, miscarriage...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.set_index('Title', inplace=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genre</th>
      <th>Director</th>
      <th>Actors</th>
      <th>Key_words</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>The Shawshank Redemption</th>
      <td>[crime,  drama]</td>
      <td>frankdarabont</td>
      <td>[timrobbins, morganfreeman, bobgunton]</td>
      <td>[two, imprisoned, men, bond, number, years, fi...</td>
    </tr>
    <tr>
      <th>The Godfather</th>
      <td>[crime,  drama]</td>
      <td>francisfordcoppola</td>
      <td>[marlonbrando, alpacino, jamescaan]</td>
      <td>[aging, patriarch, organized, crime, dynasty, ...</td>
    </tr>
    <tr>
      <th>The Godfather: Part II</th>
      <td>[crime,  drama]</td>
      <td>francisfordcoppola</td>
      <td>[alpacino, robertduvall, dianekeaton]</td>
      <td>[early, life, career, vito, corleone, 1920s, n...</td>
    </tr>
    <tr>
      <th>The Dark Knight</th>
      <td>[action,  crime,  drama]</td>
      <td>christophernolan</td>
      <td>[christianbale, heathledger, aaroneckhart]</td>
      <td>[menace, known, joker, emerges, mysterious, pa...</td>
    </tr>
    <tr>
      <th>12 Angry Men</th>
      <td>[crime,  drama]</td>
      <td>sidneylumet</td>
      <td>[martinbalsam, johnfiedler, leej.cobb]</td>
      <td>[jury, holdout, attempts, prevent, miscarriage...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col!= 'Director':
            words = words + ' '.join(row[col]) + ' '
        else:
            words = words + row[col]+ ' '
    row['bag_of_words'] = words
    
df.drop(columns = [col for col in df.columns if col != 'bag_of_words'])
```

    /var/folders/4y/3qvgxp5d62l2_0f4sx2hgqdr0000gn/T/ipykernel_90801/3985703523.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['bag_of_words'] = ''





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bag_of_words</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>The Shawshank Redemption</th>
      <td>crime  drama frankdarabont timrobbins morganfr...</td>
    </tr>
    <tr>
      <th>The Godfather</th>
      <td>crime  drama francisfordcoppola marlonbrando a...</td>
    </tr>
    <tr>
      <th>The Godfather: Part II</th>
      <td>crime  drama francisfordcoppola alpacino rober...</td>
    </tr>
    <tr>
      <th>The Dark Knight</th>
      <td>action  crime  drama christophernolan christia...</td>
    </tr>
    <tr>
      <th>12 Angry Men</th>
      <td>crime  drama sidneylumet martinbalsam johnfied...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>The Lost Weekend</th>
      <td>drama  film-noir billywilder raymilland janewy...</td>
    </tr>
    <tr>
      <th>Short Term 12</th>
      <td>drama destindanielcretton brielarson johngalla...</td>
    </tr>
    <tr>
      <th>His Girl Friday</th>
      <td>comedy  drama  romance howardhawks carygrant r...</td>
    </tr>
    <tr>
      <th>The Straight Story</th>
      <td>biography  drama davidlynch sissyspacek janega...</td>
    </tr>
    <tr>
      <th>Slumdog Millionaire</th>
      <td>drama dannyboyle,loveleentandan devpatel saura...</td>
    </tr>
  </tbody>
</table>
<p>250 rows × 1 columns</p>
</div>




```python
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])
```


```python
c= count_matrix.todense()
c
```




    matrix([[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]])




```python
print(count_matrix[0,:])
```

      (0, 584)	1
      (0, 768)	1
      (0, 1011)	1
      (0, 2678)	1
      (0, 1810)	1
      (0, 306)	1
      (0, 2765)	1
      (0, 1269)	1
      (0, 1733)	1
      (0, 311)	1
      (0, 1899)	1
      (0, 2950)	1
      (0, 969)	1
      (0, 2481)	1
      (0, 888)	1
      (0, 2174)	1
      (0, 59)	1
      (0, 519)	1
      (0, 655)	1



```python
#Generate cosine similarity matrix 
cos_sim = cosine_similarity(count_matrix, count_matrix)
cos_sim
```




    array([[1.        , 0.15789474, 0.13764944, ..., 0.05263158, 0.05263158,
            0.05564149],
           [0.15789474, 1.        , 0.36706517, ..., 0.05263158, 0.05263158,
            0.05564149],
           [0.13764944, 0.36706517, 1.        , ..., 0.04588315, 0.04588315,
            0.04850713],
           ...,
           [0.05263158, 0.05263158, 0.04588315, ..., 1.        , 0.05263158,
            0.05564149],
           [0.05263158, 0.05263158, 0.04588315, ..., 0.05263158, 1.        ,
            0.05564149],
           [0.05564149, 0.05564149, 0.04850713, ..., 0.05564149, 0.05564149,
            1.        ]])




```python
# creating a series for the movie titles so they are associated with an ordered numerical list
indices = pd.Series(df.index)
indices[:20]
```




    0                              The Shawshank Redemption
    1                                         The Godfather
    2                                The Godfather: Part II
    3                                       The Dark Knight
    4                                          12 Angry Men
    5                                      Schindler's List
    6         The Lord of the Rings: The Return of the King
    7                                          Pulp Fiction
    8                                            Fight Club
    9     The Lord of the Rings: The Fellowship of the Ring
    10                                         Forrest Gump
    11       Star Wars: Episode V - The Empire Strikes Back
    12                                            Inception
    13                The Lord of the Rings: The Two Towers
    14                      One Flew Over the Cuckoo's Nest
    15                                           Goodfellas
    16                                           The Matrix
    17                   Star Wars: Episode IV - A New Hope
    18                                                Se7en
    19                                It's a Wonderful Life
    Name: Title, dtype: object




```python
# Function that takes in movie title as input and returns the top 10 recommended movies
def recommendations(title, cos_sim=cos_sim):
    
    recommended_movies = []
    
    # getting the index of the movie that matches the title
    idx= indices[indices == title].index[0]
    
    # creating a Series with the similarity score in descending order 
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending=False)
    
    #getting the indexes of the 10 most similar movies 
    top_10_indexes = list(score_series.iloc[1:11].index)
    print(top_10_indexes)
    
    #populating the list with the title of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
        
    return recommended_movies
```

### Now, for the fun part, I will see the recommendation my project gives me for some of the movies I like 

- Interstellar
- Snatch
- Blood Diamond
- The Godfather
- Fight Club


```python
recommendations('Interstellar')
```

    [40, 222, 55, 237, 74, 219, 12, 167, 199, 69]





    ['The Prestige',
     'The Martian',
     'Aliens',
     'The Revenant',
     '2001: A Space Odyssey',
     'The Avengers',
     'Inception',
     'The Truman Show',
     'Guardians of the Galaxy',
     'Eternal Sunshine of the Spotless Mind']




```python
recommendations('Snatch')
```

    [54, 115, 109, 218, 234, 151, 1, 125, 214, 43]





    ['Once Upon a Time in America',
     'The Wolf of Wall Street',
     'Lock, Stock and Two Smoking Barrels',
     'The Killing',
     'Blood Diamond',
     'Butch Cassidy and the Sundance Kid',
     'The Godfather',
     'The Big Lebowski',
     'Arsenic and Old Lace',
     'The Great Dictator']




```python
recommendations('Blood Diamond')
```

    [237, 34, 201, 98, 140, 239, 232, 147, 161, 63]





    ['The Revenant',
     'The Departed',
     'Jaws',
     'The Gold Rush',
     'Shutter Island',
     'The Manchurian Candidate',
     'JFK',
     'Stand by Me',
     'What Ever Happened to Baby Jane?',
     'Requiem for a Dream']




```python
recommendations('The Godfather')
```

    [2, 83, 128, 226, 100, 15, 123, 76, 110, 66]





    ['The Godfather: Part II',
     'Scarface',
     'Fargo',
     'Rope',
     'On the Waterfront',
     'Goodfellas',
     'Cool Hand Luke',
     'Baby Driver',
     'Casino',
     'A Clockwork Orange']




```python
recommendations('Fight Club')
```

    [137, 246, 123, 85, 135, 245, 243, 167, 53, 26]





    ['Gone Girl',
     'Short Term 12',
     'Cool Hand Luke',
     'Good Will Hunting',
     'Into the Wild',
     'The Lost Weekend',
     'Big Fish',
     'The Truman Show',
     'American Beauty',
     'American History X']

