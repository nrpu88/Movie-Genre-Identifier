import urllib2
import requests
import json
import imdb
import time
import itertools
import wget
import os
import tmdbsimple as tmdb
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import unicodecsv as csv
########################################MOST IMPORTANTLY USE UTF-8 ENCODING EVERYWHERE##################################







tmdb.API_KEY = api_key #This sets the API key setting for the tmdb object
search = tmdb.Search() #this instantiates a tmdb "search" object which allows your to search for the movie
def get_movie_genres_tmdb(movie):
    response = search.movie(query=movie)
    id=response['results'][0]['id']
    movie = tmdb.Movies(id)
    genres=movie.info()['genres']
    return genres




##################################################################################

#NEED TO PUT THIS IN A LOOP TO OBTAIN THE DATASET(SNO,TITLE,PLOT,GENRE) IN A CSV []

all_movies=tmdb.Movies()
top1000_movies=[]
print('Pulling movie list, Please wait...')
for i in range(1,999):#Change the number of movies you want 

	movies_on_this_page=all_movies.popular(page=i)['results']
	top1000_movies.extend(movies_on_this_page)

len(top1000_movies)

print(top1000_movies[0])



sno = 1
NoOfActionMovies = 0
NoOfComedyMovies = 0
NoOfThrillerMovies = 0
NoOfDramaMovies = 0
NoOfScienceFictionMovies = 0

GenreSet = {"Action","Comedy","Romance","Horror","Drama"}

with open('romcom.csv','w') as data_file:
	
	writer = csv.writer(data_file)

	writer.writerow(['sno','title','plot','genre'])
		
	for i in range(1,7000):##change the number of movies you want


		overview = top1000_movies[i]['overview']
		punctuations = '''!()-[]{};:'",<>./?@#$%^&*_~'''
		no_punct = ''
		my_str = overview
		for char in my_str:
			if char not in punctuations:
				no_punct = no_punct + char

		overview = no_punct

		genre = get_movie_genres_tmdb(top1000_movies[i]['original_title'])
		print(len(genre))
		if len(genre) == 0:
			continue
		print(genre[0])
		maj_genre = genre[0]['name']

		if maj_genre in GenreSet:
			if maj_genre=="Action":
				if NoOfActionMovies<1200:					#Change here the total limit for the NoOfActionMovies in dataset
					NoOfActionMovies=NoOfActionMovies + 1
					writer.writerow([sno,top1000_movies[i]['original_title'],overview,maj_genre ])

			if maj_genre=="Comedy":							#Change here the total limit for the NoOfComedyMovies in dataset
				if NoOfComedyMovies<1200:
					NoOfComedyMovies=NoOfComedyMovies + 1
					writer.writerow([sno,top1000_movies[i]['original_title'],overview,maj_genre ])

			if maj_genre=="Romance":						#Change here the total limit for the NoOfThrillerrMovies in dataset
				if NoOfThrillerMovies<1200:
					NoOfThrillerMovies=NoOfThrillerMovies + 1
					writer.writerow([sno,top1000_movies[i]['original_title'],overview,maj_genre ])

			if maj_genre=="Drama":							#Change here the total limit for the NoOfDramaMovies in dataset
				if NoOfDramaMovies<1200:
					NoOfDramaMovies=NoOfDramaMovies + 1
					writer.writerow([sno,top1000_movies[i]['original_title'],overview,maj_genre ])

			if maj_genre=="Horror":				#Change here the total limit for the NoOfScienceFictionMovies in dataset
				if NoOfScienceFictionMovies<1200:
					NoOfScienceFictionMovies=NoOfScienceFictionMovies + 1
					writer.writerow([sno,top1000_movies[i]['original_title'],overview,maj_genre ])

			
			
		sno = sno + 1


print("done")

#######################################################################################################
