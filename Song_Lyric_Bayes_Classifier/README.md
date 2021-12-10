# Song Lyric Bayes Classifier
## Overview
Implement a machine learning model to predict song genre based solely on lyric text. Served as team leader, implemented lyric scraping program, feature engineering, and model.  Learned skills in NLP, data scraping, and supervised learning models. Model was able to differentiate between hip-hop and country genres with over 90% accuracy.
## Comments
This was a fun project to work on, as I have a background in music (just choir in high school, nothing fancy!). It was very cool to combine this passion with computer science! It was very daunting at first to learn the ins and outs of basic natural language processing and machine learning, but it was a very fun project to work on so I was never discouraged.
## Directory Summary
- `code`: contains the code and plain text files used for this project.
    - `classifier.py`: the code for importing, engineering, and classifying the lyric data.
    - `country_lyric_text.txt`: contains the scraped lyrics for country songs used for testing.
    - `country_lyrics.csv`: contains the raw song and lyric data used for training the model.
    - `country_test_songs_artists.csv`: the country songs/artists used to scrape lyrics.
    - `Final_ROC.ipynb`: Jupyter ntebook containing the code for plotting the receiver operating characteristic curve for the binary classifier model.
    - `hiphop_lyric_text.txt`: contains the scraped lyrics for hip hop songs used for testing.
    - `hiphop_lyrics.csv`: contains the raw song and lyric data used for training the model.
    - `hiphop_test_songs_artists.csv`: the hip hop songs/artists used to scrape lyrics.
    - `lyrics_scraper.py`: the code for scraping lyrics using the test CSV files. Requires Genius API developer key.
    - `model_outputs.txt`: contains a summary of the classifier outputs.
    - `outputs.txt`: more raw version of model outputs used for `Final_ROC.ipynb`.
- `documents`: contains project documents like the project report, presentation, etc.
    - `Country song (3).jpg`: contains a word cloud visualizing the most common words in the country songs analyzed.
    - `Final Project Presentation.pptx`: the slide deck for the project presentation.
    - `Final Project Proposal.pdf`: project proposal document.
    - `Final Project Report.pdf`: project final report.
    - `hiphop_wordcloud.jpg`: contains a word cloud visualizing the most common words in the hip hop songs analyzed.