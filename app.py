import streamlit as st
import numpy as np
from transformers import tf_top_k_top_p_filtering
from transformers.pipelines import pipeline
import tensorflow as tf
from tensorflow import keras
import re
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from tensorflow import keras
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocessing import text_preprocessing

nltk.download('vader_lexicon')


vader_model = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')
model = tf.keras.models.load_model('LSTMmodel')


def run_vader(textual_unit,
              lemmatize=False,
              parts_of_speech_to_consider=None,
              verbose=0):

    doc = nlp(textual_unit)

    input_to_vader = []

    for sent in doc.sents:
        for token in sent:

            to_add = token.text

            if lemmatize:
                to_add = token.lemma_

                if to_add == '-PRON-':
                    to_add = token.text

            if parts_of_speech_to_consider:
                if token.pos_ in parts_of_speech_to_consider:
                    input_to_vader.append(to_add)
            else:
                input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))

    if verbose >= 1:
        print()
        print('INPUT SENTENCE', sent)
        print('INPUT TO VADER', input_to_vader)
        print('VADER OUTPUT', scores)

    return scores


def vader_output_to_label(vader_output):
    compound = vader_output['compound']

    if compound < 0:
        return 'negative'
    elif compound == 0.0:
        return 'neutral'
    elif compound > 0.0:
        return 'positive'


assert vader_output_to_label(
    {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.0}) == 'neutral'
assert vader_output_to_label(
    {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.01}) == 'positive'
assert vader_output_to_label(
    {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': -0.01}) == 'negative'


def show_predict_page():
    st.title("Fake News Tester - Text Mining Project")

    st.markdown('''
    Welcome to our project website, try to paste an article in the bar below.
    ''')
    with st.expander("Example News Article", expanded=False):
        st.markdown('''
    ### Copy & paste this entire article below 
(source: https://www.reuters.com/world/americas/costa-ricans-choose-between-outsider-former-leader-presidential-vote-2022-04-03/)

Costa Ricans are casting their ballots in a run-off election Sunday, choosing between an anti-establishment outsider and former leader to be the next head of the Central American country as it grapples with debt woes and social discontent.

Final polling gave economist Rodrigo Chaves, a former longtime World Bank official, a slight lead over former President Jose Maria Figueres. Chaves had 41% of support, while Figueres was seen with 38%, with many voters still undecided, according to a poll by the University of Costa Rica published Tuesday.

Chaves, who also briefly served as finance minister for outgoing President Carlos Alvarado, came second in an initial vote in February. Seen as a renegade, he has vowed to shake up the ranks of the political elite, even pledging to use referendums to bypass Congress to bring change. read more

"If the people go out to vote, this is going to be a sweep, a tsunami," a confident Chaves said after casting his ballot on Sunday.

Figueres, whose father was also president for three separate terms, campaigned on his experience and family political legacy. He has promised to lift post-coronavirus pandemic economic growth and boost green industries in the environmentally progressive nation.

"Let's vote with joy, respecting the preferences of each person, but reinforcing our democratic system," Figueres told reporters after voting.

Going into the election, some voters said they were lukewarm on both candidates, whose political careers have been tainted by accusations of wrongdoings.

Chaves faced allegations of sexual harassment during his tenure at the World Bank, which he denied. Figueres resigned as executive director of the World Economic Forum in 2004 amid accusations in Costa Rica that he had influenced state contracts with the telecoms company Alcatel, a case that was never tried in court.

"I came because it is mandatory, but I am a little afraid of what is going to happen to the country," said Diego Ortiz, 32, a nursing assistant who voted in the morning at a polling place in Leon XIII, a poor district north of capital San Jose. "Neither of them are good candidates for me."

Another voter, David Diaz, 33, said he was not enthused about either candidate. He left his home early so he could vote by 7 a.m. in the rural town of Tacacori, about 30 km (19 miles) from San Jose.

"I see very little movement, there is a lot of apathy," said Diaz, a mechanic at a medical device factory.

Only 60% of eligible voters cast ballots in the first round, the lowest figure in decades. The margin between Chaves and Figueres, which has increasingly narrowed since Figueres led the first round, means undecided voters represent a key 18% piece of the pie that could sway the election in favor of either candidate.

"Chaves retains an edge due mostly to Figueres' relatively higher rejection rates and the weight that voters give his corruption allegations relative to Chaves' sexual harassment-related baggage," consultancy group Eurasia said in a note. "But the high level of undecided voters and very fluid voter preferences mean that Figueres could still pull out a win."

A new president must manage Costa Rica's economy, which was battered by the COVID-19 pandemic, before rebounding. About 23% of the country's population of 5.1 million live in poverty. A growing income disparity makes it one of the most unequal countries in the world and unemployment is running at almost 15%. read more

Costa Rica agreed in January 2021 to $1.78 billion in financial assistance from the International Monetary Fund. In return, the government said it would push through a raft of fiscal changes and austerity measures, but lawmakers have only passed a law to make savings on public sector workers benefits.

Polling centers opened at 6 a.m. local time (1200 GMT) and will close at 6 p.m. (0000 GMT Monday). The first results are expected after 8 p.m. local time from the headquarters of the Supreme Electoral Tribunal.

    ''')

    article_input = st.text_area(
        'News Article', placeholder='Insert text here')

    st.write('It might take some time, but try it out:')
    run = st.button('Test!')

    if run:
        article = text_preprocessing(article_input)
        # loading
        tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))

        text = tokenizer.texts_to_sequences([article])

        input_text = pad_sequences(text, padding='post', maxlen=400)

        model = keras.models.load_model('./LSTMmodel')

        pred = model.predict(input_text)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        prediction = pred.tolist()[0][0]

        summarizer = pipeline('summarization',model="pegasus-cnn_dailymail")
        sentimentizer = pipeline('sentiment-analysis', model="siebert/sentiment-roberta-large-english")

        summary = summarizer(article_input, truncation=True)
        sentiment = sentimentizer(article_input, truncation=True)
        VADER_out = run_vader(article_input)
        VADER_sentiment = vader_output_to_label(VADER_out)

        st.markdown(
            f'''
            ### Article Summary:
            {summary[0]['summary_text']}

            ### Article Sentiment Rating:
            #### Transformer model :
            {sentiment[0]['label'].lower().capitalize()}
            #### VADER model:
            {VADER_sentiment.capitalize()}

            ### LSTM Classifier Prediction:
            '''
        )

        if prediction > 0:
            st.write(f'Assigned label: {int(prediction)}')
            st.write('According to this LSTM model, this might be fake news...')
            st.warning('This might be fake news!')
            st.snow()
        else:
            st.write(f'Label: {int(prediction)}')
            st.write(
                'According to this LSTM model this article seems to contain real news!')
            st.warning('This might be real news!')
            st.balloons()


show_predict_page()
