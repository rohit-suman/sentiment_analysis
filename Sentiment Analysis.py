#!/usr/bin/env python
# coding: utf-8

# #### BLACKCOFFER PROJECT

# In[2]:


# Importing Libraries

import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.tokenize import regexp_tokenize
import gensim
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
tqdm.pandas()


# In[3]:


# Importing Output Data Structure excel file for links

df = pd.read_excel('C:/Users/DELL/Desktop/Study Materials/6_Blackcoffer Project/Output Data Structure.xlsx')


# In[4]:


# Place all the URLs in list

url_list = df['URL'].tolist()


# In[7]:


# Finding Out sentiment_score for Sentiment_Analysis   
    
def sentiment_score(x):

    ## Data from website:    
    headers = {
        'User-Agent': 'BlackCoffer Project'
    }

    BlackC = requests.get(str(x), headers=headers)
    content = BlackC.content

    soup = BeautifulSoup(content, 'lxml')
    post = soup.find('div', class_='td-post-content')

    ## Extract data in form of text amd make lowrcase.
    c1 = post.text.lower()

    ## Tokenization - for breaking the raw text into small chunks 
    c2 = set(word_tokenize(c1))

    c3 = r"[)\(w+.[0-9]"

    c4 = set(c2) - set(c3)

    ## Stopwords
    all_stopwords = gensim.parsing.preprocessing.STOPWORDS

    c5 = c4 - all_stopwords.union(set([',','1','2','3','33','4','5', '3',':', 'a.m.', 'ai','t','—for', 'he/she', '’','“','”']))

    a = list(str(c5).split())

    ## initialise stemmer & lemmatizer
    pst = PorterStemmer()
    lt = WordNetLemmatizer()

    ### sample data frame
    df = pd.DataFrame({'senten': a})

    #### apply here
    df['senten'] = df['senten'].apply(word_tokenize)
    df['senten1'] = df['senten'].apply(lambda x: ' '.join([pst.stem(y) for y in x]))
    df['senten2']  = df['senten'].apply(lambda x: ' '.join([lt.lemmatize(y) for y in x]))
    df['senten3'] = df['senten2'].str.replace(r'[^\w)(\s]+', '')

    final = list(df['senten3'])

    listToStr = ' '.join([str(elem) for elem in final])

    res = len(listToStr.split())

    number_of_sentences = sent_tokenize(c1)

    ## Personal Pronouns
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
    pronouns = pronounRegex.findall(c1)
    
    ## Word & Character Count
    word_count = 0
    char_count = 0

    split_string = c1.split()

    word_count = len(split_string)
    for word in split_string:
        char_count += len(word)
    
    # Sentiment_Analysis    
    score = SentimentIntensityAnalyzer().polarity_scores(listToStr)
    neg = score['neg']
    pos = score['pos']
    polarity_score = ((pos - neg)/(pos + neg))+0.000001 
    subjectivity_score = ((pos + neg)/len(final))+0.000001
    average_sentence_length = len(listToStr.split())/len(number_of_sentences)
    percentage_of_complex_words = syllable_count(listToStr)/len(listToStr.split())
    fog_index = 0.4 * (average_sentence_length + percentage_of_complex_words)
    Average_Number_of_Words_Per_Sentence =  len(c1.split())/len(number_of_sentences)
    Complex_Word_Count = syllable_count(listToStr)
    Word_Count = len(listToStr)
    Syllable_per_word = Complex_Word_Count/Word_Count
    Personal_pronouns = len(pronouns)
    Average_word_length = char_count/word_count
    
    
    return {'Positive Score': pos, 
            'Negative Score': neg,
            'Polarity Score': polarity_score,
            'Subjectivity Score': subjectivity_score,
            'Average Sentence Length': average_sentence_length,
            'Percentage of complex words': percentage_of_complex_words,
            'Fog Index': fog_index,
            'Average Number of Words Per Sentence': Average_Number_of_Words_Per_Sentence,
            'Complex Word Count': Complex_Word_Count,
            'Word Count': Word_Count,
            'Syllable per word': Syllable_per_word,
            'Personal Pronouns': Personal_pronouns,
            'Average Word Count': Average_word_length}
    
    
# Finding Out syllable_count for percentage_of_complex_words

def syllable_count(word):
    count = 0
    vowels = "aeiou"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
    if count == 0:
        count += 1
    return count


# In[8]:


sent = []
for i in tqdm(range(len(url_list))):
    sent.append(sentiment_score(url_list[i]))


# In[11]:


df1 = pd.DataFrame(sent)


# In[12]:


df1.to_excel('C:/Users/DELL/Desktop/Study Materials/6_Blackcoffer Project/Output Data Structure.xlsx')


#  ###### ----- Completed
