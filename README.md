# Movie-Genre-Identifier
Vectorizer based Movie Genre Identifier

We have used Hash Vectorizer for text to numbers conversion.In this aspect we have varied the number of features that need to be extracted from the text.By varying this number of features we have been able to obtain satisfactory precision values.

Coming to the ML algorithms,we have attempted to vary the parameters respective to the algorithm(for example ‘max-iter’ parameter of Stochastic Gradient Descent) namely ‘hyperparameter tuning’ and have studied the variation of these parameters on the precision metric.

We have also carried out the entire text to number conversion using “CountVectorizer” and “TfidfVectorizer” and have come to the conclusion that the “Hash Vectorizer” performs the conversion in a shorter time duration and since it allows us to have control over the size of the constructed vocabulary, it results in a higher precision score. 
