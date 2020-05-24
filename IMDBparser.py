# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:56:12 2020

@author: Mihul

"""

def _get_subcrew(crew, role): 
    import re
    import numpy as np
    if len(crew) > 1:
        subcrew = re.sub('.*:+', '', crew[role])
        subcrew = (re.sub('[\n!?]', '', subcrew))#.split(',')
    else:
        subcrew = np.nan  
        #subcrew = 'NaN'
    return subcrew #string

def make_IMDB_dataset(start_year = 2000, end_year = 2020, min_user_rating = 5.0, min_votes = 10000, **kwargs):
    
    # importing libraries
    from requests import get
    from bs4 import BeautifulSoup
    import re
    import pandas as pd
    import random # for simulating human behavior
    from time import time, sleep
    from IPython.core.display import clear_output
    from warnings import warn
    
    # get kwargs
    sort = kwargs.get('sort', 'num_votes,desc')
    country = kwargs.get('country', 'us')
    
    # variables for iterating
    pages = [str(i*50 - 49) for i in range(1,5)]
    years = [str(k) for k in range(start_year, end_year)]
    headers = {"Accept-Language": "en-US, en;q=0.5"}

    # Prepare monitor variables
    requests = 0 
    start_time = time()
    
    # Prepare a list of future DataFrames related to each page
    pages_dfs = []

    # Main loop
    for year in years:
        
        for page in pages: # 4 pages
            
            # Make a request
            url = ('https://www.imdb.com/search/title/?title_type=feature&release_date=' + year +
            '&user_rating=' + str(round(min_user_rating, 1)) +
            ',&num_votes=' + str(min_votes) + 
            ',&countries=' + country +
            '&sort=' + sort +
            '&start=' + page)
            
            response = get(url, headers = headers)
            
            # Make a random pause
            sleep(random.randint(7,16))
            
            # Monitor the requests
            requests += 1
            elapsed_time = time() - start_time
            print('Request {}: status code: {}'.format(requests, response.status_code))
            clear_output(wait = True)
            
            # Raise a warning if a status code is not 200
            if response.status_code != 200:
                warn('Request {}: status code: {}'.format(requests, response.status_code))
            
            # Stop the loop if the number of requests exceeds the maximum (4 pages per year)
            if requests > ((end_year - start_year) * 4):
                warn('Number of requests exceeds the maximum value!')
                break
            
            # Start parsing
            current_page = BeautifulSoup(response.text, 'html.parser')
            
            # ___________________ Collect the data ______________________
            
            movie_containers = current_page.find_all('div', class_ = 'lister-item mode-advanced')
            
            certificates50 = []
            runtimes50 = []
            genres50 = []
            descs50 = []
            dirs50 = []
            stars50 = []
            ratings50 = []
            
            for container in movie_containers:
                
                ''' List of independent variables:
                     - certificate (categories) - string
                     - runtime (in min) - int
                     - genre (categories) - string
                     - description (text) - text
                     - directors (categories) - string
                     - stars (categories) - string
                    
                    Target variable: Imdb rating
    
                '''
                # ___________________________ Get independent variables _________________________
                
                # certificate
                certificate = container.p.span.text
                certificates50.append(certificate)
                
                # runtime
                runtime = container.p.find('span', class_ = 'runtime').text
                runtime = int(runtime.split()[0])
                runtimes50.append(runtime)
                
                # genre
                genre = container.p.find('span', class_ = 'genre').text
                genre = (re.sub('[\n.!? ]', '', genre))#.split(',')
                genres50.append(genre)
                
                # description
                desc = (container.find('div', class_ = 'lister-item-content')).find_all('p', class_='text-muted')
                desc = desc[1].text
                descs50.append(desc)
                
                # directors and stars
                crew = container.find('p', class_ = '').text.split('|')
                dirs = _get_subcrew(crew, 0)
                dirs50.append(dirs)
                
                stars = _get_subcrew(crew, 1)
                stars50.append(stars)
                
                # ___________________________ Get target variable _________________________
                rating = float(container.strong.text)
                ratings50.append(rating)
            
            # Make a DataFrame for 1 page of 50 movies
            page_df = pd.DataFrame({'Certificate': certificates50,
                                    'Runtime': runtimes50,
                                    'Genre': genres50,
                                    'Description': descs50,
                                    'Directors': dirs50,
                                    'Stars': stars50,
                                    'Rating': ratings50})
            
            # Append a list of DFs related to each page
            pages_dfs.append(page_df)

    dataset = pd.concat(pages_dfs, ignore_index = True)
    return dataset

# __________________________________________ TESTING ______________________________________

if __name__ == '__main__':
    
    from requests import get
    from bs4 import BeautifulSoup
    import re
    import pandas as pd
    import random # for simulating human behavior
    from time import time, sleep
    from IPython.core.display import clear_output
    from warnings import warn


    
    url = 'https://www.imdb.com/search/title/?title_type=feature&release_date=2018-01-01,&user_rating=5.0,&num_votes=1000,&countries=us&sort=boxoffice_gross_us,desc'
    response = get(url)
    print(response.text[:500])
    
    html_soup = BeautifulSoup(response.text, 'html.parser')
    type(html_soup)
    
    movie_container = html_soup.find_all('div', class_ = 'lister-item mode-advanced')
    print(type(movie_container))
    print(len(movie_container))
    
    ''' List of independent variables:
         - certificate (categories) - text
         - runtime (in min) - int
         - genre (categories) - list
         - description (text) - text
         - directors (categories) - list
         - stars (categories) - list
        
        Target variable: Imdb rating
        
    '''
    # ___________________________ Get independent variables _________________________
    
    # certificate
    certificate = movie_container[0].p.span.text
    
    # runtime
    runtime = movie_container[0].p.find('span', class_ = 'runtime').text
    runtime = int(runtime.split()[0])
    
    # genre
    genre = movie_container[0].p.find('span', class_ = 'genre').text
    genre = (re.sub('[\n.!? ]', '', genre)).split(',')
    
    # description
    desc = (movie_container[0].find('div', class_ = 'lister-item-content')).find_all('p', class_='text-muted')
    desc = desc[1].text
    
    # directors and stars
    crew = movie_container[0].find('p', class_ = '').text.split('|')
    def get_subcrew(crew, role):
        subcrew = re.sub('.*:+', '', crew[role])
        subcrew = (re.sub('[\n!?]', '', subcrew))#.split(',')
        return subcrew
    dirs = get_subcrew(crew, 0)
    stars = get_subcrew(crew, 1)
    
    
    # ___________________________ Get target variable _________________________
    rating = float(movie_container[0].strong.text)
        
    
    # tracking the speed of requesting
    import random # for simulating human behavior
    from time import time
    from IPython.core.display import clear_output
    
    start_time = time.time()
    requests = 0
    for _ in range(5):
    # A request would go here
        requests += 1
        time.sleep(random.randint(1,3))    
        current_time = time.time()
        elapsed_time = current_time - start_time
        print('Request: {}; Frequency: {} requests/s'.format(requests, requests/elapsed_time)) 
    clear_output(wait = True)
    
    # warn if request status is not 200 (something wrong)
    from warnings import warn
    warn("Warning Simulation")
    
    #Serching by every year (2000 - 2020), sort by rating, first 50 pages
    pages = [i for i in range(1,5)]
    years = [k for k in range(2000, 2020)]
    headers = {"Accept-Language": "en-US, en;q=0.5"}

    test_set = make_IMDB_dataset(start_year = 2018, end_year = 2020)
