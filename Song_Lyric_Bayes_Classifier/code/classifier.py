# -----------------------------------------------------------------------------
# Florence Gurney-Cattino, Jameson Albers, Zongrui Liu
# CS 5002, Spring 2021
# Final Project: Multinomial Naive Bayes Classifier
#
# This program includes functions to import song lyrics and genre from a 
# dataset, process them, train a Bayes classifier, and output a prediction list.
# -----------------------------------------------------------------------------
import csv
import nltk
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load genre/lyrics data from .csv files. Each line in the .csv file represents
# one song, with the comma delineating "genre", "lyrics". The ouput will be a 
# list of all songs, with each song being a list with the format 
# ["Genre", "Lyrics"]
def get_lyric_data(*filenames: str) -> list:
    lyric_data = []
    for filename in filenames:
        with open(filename, newline='', encoding='utf-8') as data_file:
            data_reader = csv.reader(data_file, delimiter=',')
            for row in data_reader:
                genre, raw_lyrics = row
                raw_song = [genre, raw_lyrics]
                lyric_data.append(raw_song)
    return lyric_data


# Our training data was pulled from 
# https://data.mendeley.com/datasets/3t9vbwxgr5/3. We trained the
# classifier model with the country and hip hop genre subsets of the data.
train_data = get_lyric_data('country_lyrics.csv', 'hiphop_lyrics.csv')


# Assign each genre a numeric classifier. We used 1 for country and 0
# for hip hop. The output will be a list of songs with each song as a list
# formatted as [Class, "Lyrics"]
def genre_to_class(data_list: list) -> None:
    for song in data_list:
        if song[0] == 'country':
            song[0] = 1
        else:
            song[0] = 0

# Run the function with our training data
genre_to_class(train_data)


# The training dataset we used alread had the lyrics as a string of lemmatized
# words separated by white space. To put it into a "bag of words" format, we 
# split the long lyrics string into a list of strings, each string being a word.
# We then added each word to a dictionary data type, with the key being the word
# and the value being the number of times the word occurs in the song. The
# output will be a list of songs, with each song being in the format
# [Class, {"Word": Occurence}]
def raw_lyrics_to_bag(classified_data: list) -> None:
    for song in classified_data:
        word_list = song[1].split()
        bag_dict = dict()
        for word in word_list:
            if word in bag_dict:
                bag_dict[word] += 1
            else:
                bag_dict[word] = 1
        song[1] = bag_dict


# Run the function with our training data
raw_lyrics_to_bag(train_data)


# Now we make a set of all the words in our training data set. We need this
# later to correctly feature engineer our test data.
def get_total_words(bow_data: list) -> None:
    function_list = bow_data.copy()
    words_output = set()
    for song in function_list:
        for word in song[1]:
            words_output.add(word)
    return words_output


# Get the set of words from our training data
model_words = get_total_words(train_data)


# Each song in our training data is still in the form of [Classifier, {Words}].
# To train a multinomial naive Bayes model, we need to split this data into
# two lists: one list of classes and one list of dictionaries. The
# following function returns a list of classes representing the genre of each
# song in our training data set.
def get_class_vector(bow_data: list) -> list:
    function_list = bow_data.copy()
    class_vector = []
    for song in function_list:
        class_vector.append(song[0])
    return class_vector


# This function returns a list of dictionaries representing the lyrics of each
# song in our training data set.
def get_bag_vector(bow_data: list) -> list:
    function_list = bow_data.copy()
    bow_vector = []
    for song in function_list:
        bow_vector.append(song[1])
    return bow_vector


# Run the class list function with the training data
data_classes = get_class_vector(train_data)

# Run the feature list function with the training data
data_bags = get_bag_vector(train_data)


# The DictVectorizer object from sklearn turns a list of dictionaries into a 
# NumPy array vector that numerically represents the bag-of-words in a format
# that can be fed into Bayes' Theorem to make predictions. Each feature vector
# contains the number of occurences of every word in our test data.
dv = DictVectorizer(sparse=False)
X = dv.fit_transform(data_bags)

# We must also format our test data classes list as a NumPy array for use with
# the multinomial naive Bayes classifier.
Y = np.array(data_classes)

# Now, we can generate a Bayes classifier and train it with our feature and 
# class vectors to make predictions!
mnb = MultinomialNB()
mnb.fit(X,Y)


# We chose to use the Billboard Hot 100 Country 2020 and Billboard Hot 100
# Hip Hop/R&B 2020 charts as a test for our multinomial naive Bayes classifier.
# The song and artist information was manually transcribed from the Billboard
# website and fed into the Genius API to scrape the Genius database to get the
# lyrics for each song (see the companion lyrics_scraper.py program). We now
# need to open those text files and split the text into lists of songs.
with open('country_lyric_text.txt', encoding='utf-8') as country_test_file:
    raw_country_test_list = country_test_file.read().split('<|EndOfSong|>')

with open('hiphop_lyric_text.txt', encoding='utf-8') as hiphop_test_file:
    raw_hiphop_test_list = hiphop_test_file.read().split('<|EndOfSong|>')


# We now have 2 lists of songs (one country and one hip hop), each song
# formatted as a single string. We still have to clean up and feature engineer
# the lyrics for use with our model. The next step converts the text of each 
# song to lowercase.
def convert_to_lowercase(song_lyric_list):
    output_list = []
    for item in song_lyric_list:
        output_list.append(item.lower())
    return output_list

# Convert both test data lists to lowercase
lowercase_country = convert_to_lowercase(raw_country_test_list)
lowercase_hiphop = convert_to_lowercase(raw_hiphop_test_list)


# Now we need to tokenize each song. This will turn the single string of text
# for each song into a list of strings, each string being a single word.
def tokenize (song_lyric_list):
    output_list = []
    for song in song_lyric_list:
        tokenized_song = nltk.word_tokenize(song)
        output_list.append(tokenized_song)
    return output_list


# Tokenize both song lists
tokenized_country = tokenize(lowercase_country)
tokenized_hiphop = tokenize(lowercase_hiphop)


# We must also lemmatize each song to eliminate similar variations and
# inflections of root words (i.e. "rocks" becomes "rock", "better" becomes 
# "good").
def lemmatize(song_token_list):
    output_list = []
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for song in song_token_list:
        lemmatized_song = []
        for token in song:
            lemmatized_song.append(lemmatizer.lemmatize(token))
        output_list.append(lemmatized_song)
    return output_list


# Lemmatize both lists
lemmatized_country = lemmatize(tokenized_country)
lemmatized_hiphop = lemmatize(tokenized_hiphop)


# We also need to eliminate stop words (common words like "I, you, can, etc.")
# that do not carry any useful substance or sentiment. We used the NLTK stop
# words corpus.
stop_words = set(nltk.corpus.stopwords.words('english'))

def remove_stop_words(lemmatized_list, stop_word_set):
    output_list = []
    for song in lemmatized_list:
        new_song = []
        for word in song:
            if word not in stop_word_set:
                new_song.append(word)
        output_list.append(new_song)
    return output_list


# Remove stop words from our data lists
no_stop_country = remove_stop_words(lemmatized_country, stop_words)
no_stop_hiphop = remove_stop_words(lemmatized_hiphop, stop_words)


# Now we must do the final preparation of our data. The following function
# converts each list to a bag-of-words format and adds a genre class (1 for
# country, 0 for hip hop). We also remove all words that aren't in the set of
# words from our training data, and add any words from our training data that
# were missing in the test data.
def data_prep(country_list, hiphop_list, model_word_set):
    cleaned_data = []
    for song in country_list:
        bag_dict = dict()
        for word in model_word_set:
            if word in song:
                if word in bag_dict:
                    bag_dict[word] += 1
                else:
                    bag_dict[word] = 1
            else:
                bag_dict[word] = 0
        cleaned_data.append([1, bag_dict])       
    for song in hiphop_list:
        bag_dict = dict()
        for word in model_word_set:
            if word in song:
                if word in bag_dict:
                    bag_dict[word] += 1
                else:
                    bag_dict[word] = 1
            else:
                bag_dict[word] = 0
        cleaned_data.append([0, bag_dict])
    return cleaned_data


# Prep our data with the function
test_data = data_prep(no_stop_country, no_stop_hiphop, model_words)


# Now we separate the test data into separate song and feature lists to feed
# into the classifier model for prediction.
test_features = []
for song in test_data:
    test_features.append(song[1])

test_classes_actual = []
for song in test_data:
    test_classes_actual.append(song[0])


# The test feature list still needs to be transformed into a NumPy feature
# vector for use with the Multinomial NB object.
X_test = dv.fit_transform(test_features)


# Now we can predict the class of our test songs with our model!
results = mnb.predict(X_test)


# The predictions are formatted in a NumPy array, so we turn it into a list
# so it can be used with other Python programs.
results_list = results.tolist()


# We also need the probability calculations for each class prediction for every
# song in our test data to generate a ROC curve for the model.
probabilities = mnb.predict_proba(X_test).tolist()

print(mnb.classes_)

# We will write these to a text file to save and share the results.
with open('outputs.txt', 'w', encoding='utf-8') as outputs_file:
    outputs_file.writelines([
        'Test Data Actual Classes:\n',
        str(test_classes_actual) + '\n\n',
        '',
        'Test Data Predicted Classes:\n',
        str(results_list) + '\n\n',
        ''
        'Numeric Probability Values:\n',
        str(probabilities)
    ])

