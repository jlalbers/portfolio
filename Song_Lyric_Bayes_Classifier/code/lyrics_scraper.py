# -----------------------------------------------------------------------------
# Florence Gurney-Cattino, Jameson Albers, Zongrui Liu
# CS 5002, Spring 2021
# Final Project: Lyrics Scraper
#
# This program uses the lyricsgenius package and the Genius API to retrieve
# song lyrics from the Genius lyrics database.
# -----------------------------------------------------------------------------

import lyricsgenius as lg 
import csv

# Import the files containing the top 100 country and hip hop songs so we can
# scrape the lyrics
hiphop_songs_artists_file = open('hiphop_test_songs_artists.csv')
country_songs_artists_file = open('country_test_songs_artists.csv')


# Read the files and return the information as a list formatted as 
# ["Song", "Artist"]
def get_songs_artists(file):
    reader = csv.reader(file, delimiter=',', quotechar='"')
    output_list = []
    for line in reader:
        song, artist = line
        output_list.append([song, artist])
    return output_list


# Read the country and hip hop files
hiphop_song_artist = get_songs_artists(hiphop_songs_artists_file)
country_song_artist = get_songs_artists(country_songs_artists_file)

# Close the files
hiphop_songs_artists_file.close()
country_songs_artists_file.close()


# Create files we can use to write the lyrics from the Genius API
hiphop_lyrics_file = open('hiphop_lyric_text.txt', 'w', encoding='utf-8')
country_lyrics_file = open('country_lyric_text.txt', 'w', encoding='utf-8')


# Open the Genius API
genius = lg.Genius(
    '<Insert-Genius-API-Client-Key>', 
    skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], 
    remove_section_headers=True, verbose=True)


# Use the ["Song", "Artist"] lists to access the API and write the lyrics to
# files
def get_lyrics(arr, output_file):
    for item in arr:
        try:
            song = genius.search_song(item[0], item[1])
            s = song.lyrics
            output_file.write(s + '\n')
            output_file.write('\n<|EndOfSong|>\n\n')
        except:
            print('Unable to get song ' + str(c+1))

# Get the lyrics for our hip hop and country song lists
get_lyrics(hiphop_song_artist, hiphop_lyrics_file)
get_lyrics(country_song_artist, country_lyrics_file)


# Close the files
hiphop_lyrics_file.close()
country_lyrics_file.close()
