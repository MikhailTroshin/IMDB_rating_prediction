# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:03:32 2020

@author: Mihul

Preprocessing module for cleaning and preparing the raw data from IMDB website
"""

# importing libraries
import pandas as pd
import numpy as np

# ________________________ MAIN FUNCTIONS __________________________

def clean_text(text, ** kwargs):
    """
    Function cleans text for further 'Bag of words' or other NLP algorithms. 
    
    Parameters
    ----------
    text : string
        DESCRIPTION.
        
    ** kwargs : stemmer - string
        - 'Porter' switches on Porter stemming algorithm
        - 'Snowball' switches on Snowball stemming algorithm

    Returns
    -------
    String

    """
    import re
    # getting kwargs
    stemmer = kwargs.get('stemmer', None)
    
    # preprocess text
    result = re.sub('[^a-zA-Z]', ' ', text)
    result = result.lower()
    result = result.split()
    # stem if needed
    if stemmer == 'Porter':
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        ps = PorterStemmer()
        result = [ps.stem(word) for word in result if not word in set(stopwords.words('english'))]
    elif stemmer == 'Snowball':
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.snowball import SnowballStemmer
        ss = SnowballStemmer(language='english')
        result = [ss.stem(word) for word in result if not word in set(stopwords.words('english'))]
    result = ' '.join(result)
    return result
    

def get_multidummies(data, sep, **kwargs):
    """
    Function makes a dummy-variable table from column, that contains a 
    'sep'-separated strings, each of which has to be processed as unique variable.

    Parameters
    ----------
    data : 
        pd.Series
        DESCRIPTION.
    sep : 
        String
        Separator for values in cells of input data
    kwargs : 
        n_values: int
            Number of cells for working with.
            To use all dataset enter '0' value.
            If we have a lot of repeating values, it is 
            no need to search through all set.
        drop_first: bool
            Drop the first column of resulting dataset or not.
        reduce: string
            The method for reducing the size of resulting set.

    Returns
    -------
    pd.DataFrame
        DESCRIPTION.

    """
    # getting kwargs
    #column = kwargs.get('columns', 0)
    n_values = kwargs.get('n_values', 100)
    drop_first = kwargs.get('drop_first', False)
    #reduce = kwargs.get('reduce', 'popularity')
    if n_values > len(data) or n_values == 0:
        n_values = len(data)
    
    # iterating through n_values of a data to get all possible variants
    whole_list = []
    for index, value in data[:n_values].items():
        temp_list = value.split(sep)
        for item in temp_list:
            if item not in whole_list:
                whole_list.append(item)
            
    # creating new empty dataframe            
    new_data = pd.DataFrame(np.zeros([n_values, len(whole_list)]), index = data[:n_values].index, columns = whole_list) 

    # iterating through all original values and setting corresponding non-zero elements
    for index, value in data[:n_values].items():
        temp_list = value.split(sep)
        for item in temp_list:
            new_data.at[index, item] = 1
            
    return new_data.iloc[:, 1:] if drop_first else new_data


def reduce_by_min_count(data_to_reduce, target_cols, min_row_count = 1):
    
    """
    Function tries to reduce the 'data_to_reduce' by deleting columns from
    'target_cols' list only if at least 'min_row_count' of values remains at
    each row after delete.

    Parameters
    ----------
    data_to_reduce : 
        DataFrame
        DESCRIPTION.
        Dummy-DataFrame with 0/1 values.
    target_cols : TYPE 
    list
        DESCRIPTION.
        List of column names that we will try to delete
    min_row_count : TYPE int, optional
        DESCRIPTION. The default is 1.

    Returns
    DataFrame
    -------
    None.

    """
    
    from copy import copy
    
    data = copy(data_to_reduce)
    # iterate through all columns from 'target_cols' list
    for column in target_cols:
        
        # check if the current column name exists in input 'data_to_reduce'
        if column in data.columns:
            delete = True
            remaining = data.drop(columns = [column])
            
            # alternative           
            if True in (remaining.sum(axis = 0) == 0).values:
                delete = False
            '''
            # iterate through all rows of input 'data_to_reduce'
            for row in range(len(data)):
                min_count_in_row = remaining.loc[row, :].sum()
                
                if min_count_in_row == 0: # there are no rows witout any 
                    #non-zero element in it after droping our current column
                    delete = False
                    break '''
            if delete:
                #print(column)
                del data[column]
        else:
            print("Column '{}' does not exist in input data_to_reduce".format(column))
            
    return data
    
    
       
#___________________ Testing __________________


if __name__ == '__main__':

    import matplotlib.pyplot as plt 
    import seaborn as sns
    import IMDBparser as parser
    
    
    #%matplotlib inline
    sns.set(color_codes = True)
    
    # making dataset from 2000 to 2019 years
    dataset = parser.make_IMDB_dataset(start_year = 2000, end_year = 2020)
    dataset.isnull().sum()
    # replace 'NaN' strings with np.nan value
    dataset.replace(to_replace = 'NaN', value = np.nan, inplace = True)
    dataset.isnull().sum()
    # delete rows with nan values
    dataset = dataset.dropna()
    # saving datset
    dataset.to_csv('dataset_00_19.csv')
    
    dataset = pd.read_csv('dataset_00_19.csv')
    testset = pd.read_csv('testset.csv')
    testsplit = testset['Stars'][:10]
    testsplit = get_multidummies(testsplit, sep = ',')
    
    # explore the data
    from pandas_profiling import ProfileReport
    profile = ProfileReport(testset)
    profile.to_file('data_profile.html')
    
    plt.plot(testset['Rating'])
    plt.show()
    
    sns.distplot(testset['Rating'], rug = True)
    sns.kdeplot(dataset['Rating'], shade = True)
    
    # count unique values of some columns
    
    def make_list_of_uniques(column):
        list_of_uniques = []
        for index, string in column.items():
            for value in string.split(','):
                if value not in list_of_uniques:
                    list_of_uniques.append(value)
        return list_of_uniques
    
    # Genre
    list_of_Genres = make_list_of_uniques(testset['Genre'])       
    print(len(list_of_Genres)) # 20 unique genres
    print(len(testset), len(list_of_Genres), len(testset['Genre'].unique())) # 106 different combinations - much more then unique genres
    # solution: encode by unique values - func get_multidummies
    
    # Directors
    list_of_dirs = make_list_of_uniques(testset['Directors'])
    print(len(testset), len(list_of_dirs), len(testset['Directors'].unique())) # numbers are close
    # simple dummy-encoding, cos number of unique values is greater then the number of combinations
    
    # Actors
    list_of_stars = make_list_of_uniques(testset['Stars'])
    print(len(testset), len(list_of_stars), len(testset['Stars'].unique())) # list of unique stars is 3 times larger then list of combinations (equals to len(testset))
    # enode unique actors with reducing by popularity
    
    # Description
    # clean texts and vectorize with bag of words or Tf-idf
    
           
    # get_multidummies         
    sep = ','        
    whole_list = []
    for index, value in testsplit.items():
        temp_list = value.split(sep)
        for item in temp_list:
            if item not in whole_list:
                whole_list.append(item)
                
    # create new empty dataframe            
    new_data = pd.DataFrame(np.zeros([len(testsplit), len(whole_list)]), index = testsplit.index, columns = whole_list) 
    
    
    # iterate through all original values and set corresponding non-zero elements
    for index, value in testsplit.items():
        temp_list = value.split(sep)
        for item in temp_list:
            new_data.at[index, item] = 1
    
    test_new_data = get_multidummies(testsplit, ',', drop_first = True)
    
    
    # reduce_by_popularity
    min_films = 1
    stars_data = get_multidummies(dataset['Stars'], sep = ',', n_values = 0) 
    
    # sort by increasing popularity
    actors_by_count = stars_data.sum(axis = 0).sort_values(ascending = True)
    plt.hist(actors_by_count, bins = [1,2,3,4])
    
    # get all actors with n_films = 1
    loosers = stars_data.sum(axis = 0) == 1
    loosers = loosers[loosers == True].index #list of over 3000 actors - huge potential for reducing
    
    # iterating throurh all loosers and trying to drop this actor's column from the original actors data
    '''for actor in loosers:
        remaining_actors = stars_data.drop(columns = [actor])
        for film in range(len(stars_data)):
            if remaining_actors.loc[film, :].sum()'''
    
    loosers_ = stars_data.loc[:, loosers]
      
    testsplit = testset['Stars'][:10]
    testsplit = get_multidummies(testsplit, sep = ',')
    testsplit.to_csv('testsplit.csv')
    del testsplit['Unnamed: 0']
    
    loosers = testsplit.columns[:10]
    actor = loosers[0]
    
    
    #size before = 10 x 39
    for actor in loosers:
        delete = True
        remaining_actors = testsplit.drop(columns = [actor])
        for film in range(len(testsplit)):
            min_actors_in_film = remaining_actors.loc[film, :].sum()
            if min_actors_in_film == 0: # there ara no flims witout any actor in it after droping our current actor
                delete = False
                break
        if delete:
            print(actor)
            del testsplit[actor]           
    #size after = 10 x 31 
            
    # reduce_by_popularity_test - all is OK!
    #size before = 10 x 39      
    testsplit2 = reduce_by_min_count(testsplit, loosers)
    #size after = 10 x 31
    
    # clean_text  - all is OK!
    test_text = dataset['Description'][2]
    cleaned_no_stem = clean_text(test_text)
    cleaned_porter = clean_text(test_text, stemmer = 'Porter')
    cleaned_snow = clean_text(test_text, stemmer = 'Snowball')
