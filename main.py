import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from itertools import chain
from nltk import FreqDist
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

st.set_page_config(layout="wide")

st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: rgba(28, 131, 225, 0.1);
   border: 1px solid rgba(28, 131, 225, 0.1);
   padding: 5% 5% 5% 10%;
   border-radius: 5px;
   color: rgb(30, 103, 119);
   overflow-wrap: break-word;
}
/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: black;
}
</style>
"""
, unsafe_allow_html=True)


# load data to the app 
review_data = pd.read_csv("data/app_data.csv")
review_data.drop_duplicates(subset=['review_id','date','user_id'],inplace=True) 
review_data.set_index('Unnamed: 0')
business_name = np.unique(review_data['name'])
postal_code = np.unique(review_data['postal_code'])

# set sidebar for this app
with st.sidebar:
    postal_select = st.sidebar.selectbox(
    "Choose postal_code",
    postal_code
)
    business_select = st.sidebar.selectbox(
    "Choose Restaturant",
    np.unique(review_data['name'][review_data['postal_code']==postal_select])
)
    st.write("Contact Info : zhao468@wisc.edu")


#  sentiment scores
avg_stars = np.round(review_data['stars_x'][review_data['name']==business_select].mean(),2)
avg_rating = np.round(review_data['stars_y'][review_data['name']==business_select].mean(),2)
avg_senti = review_data['sentiment_score'][review_data['name']==business_select].mean()

#review count
count_rev = review_data['text'][review_data['name']==business_select].count()


# set header for this app
header = st.container()
with header:
	st.title('Recommendation for Chinese Restaurants in Philadelphia')
	st.write('**You can choose the business name in the left side bar and gain some analysis based on that.**')

tab1, tab3, tab4 = st.tabs(["Basic Information", "Sentiment Analysis", "Dishes Recommendation"])

# location of the business restarant
lat = np.unique(np.array(review_data['latitude'][review_data['name']==business_select]))
lon = np.unique(np.array(review_data['longitude'][review_data['name']==business_select]))
map_data = pd.DataFrame({'lat':lat,'lon':lon})

with tab1:
	col1, col2, col3 = st.columns(3)
	col1.metric("Business Stars", avg_rating )
	col2.metric("Average Stars from Customers", avg_stars)
	col3.metric("Number of reviews",count_rev)
	st.map(map_data)

# category sentiment score plot
from itertools import chain

# return list from series of \n-seperated lines
def chainer(s):
    return list(chain.from_iterable(s.str.split('\n')))

# calculate lengths of splits
lens = review_data['text'].str.split('\n').map(len)

# create new data frames, repeating or chaining as appropriate
review_segment = pd.read_csv("data/segment_data.csv")
service_data = review_segment[(review_segment['text'].str.contains('service|wait|serve'))&(review_data['name']==business_select)]
food_data = review_segment[(review_segment['text'].str.contains('food|taste|dish|cook'))&(review_data['name']==business_select)]
service_score = service_data['sentiment_score'].groupby(service_data['business_name']).median()
food_score = food_data['sentiment_score'].groupby(food_data['business_name']).median()
price_data = review_segment[(review_segment['text'].str.contains('price'))&(review_data['name']==business_select)]
price_score = price_data['sentiment_score'].groupby(price_data['business_name']).median()
X = ["Service","Food","Price"]
y_service = service_score.values[0] if service_score.values.size !=0 else 0
y_food = food_score.values[0] if food_score.values.size !=0 else 0
y_price = price_score.values[0] if price_score.values.size !=0 else 0
y = [y_service, y_food, y_price]
df = pd.DataFrame({"Category":X,"Sentiment Score":y})
fig_1,ax = plt.figure(figsize=(5,3))
sns.barplot(data=df, x="Category", y="Sentiment Score",ax=ax)
ax.set_title('Sentiment Scores by Different Categories',fontsize = 16)
plt.tight_layout()

review_data['date_new'] = pd.to_datetime(review_data["date"]).dt.strftime('%Y-%m')
avg_all_senti = review_data['sentiment_score'].mean()
group_data = review_data.groupby(['name','date_new'])
#group_data['sentiment_score'].mean()
#sns.lineplot(y = group_data[group_data['name']=='Spice 28']['sentiment_score'], x = group_data[group_data['name']=='Spice 28']['date_new'])
#group_data[group_data['name']=='Spice 28']['sentiment_score']
group = group_data.apply(lambda a: a[:])
x = np.unique(group[group['name']==business_select]['date_new'])
y_group = group[group['name']==business_select][['date_new','sentiment_score']]
y= np.array(y_group['sentiment_score'].groupby(group['date_new']).mean())
hue = np.ones(len(y)) *avg_all_senti
#fig_2 = plt.figure(figsize=(5, 3))
#ax = sns.lineplot(y = y, x= x)
#ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
#plt.plot(x,hue)
#plt.legend(title='Sentiment Scores', loc='lower left', labels=['Scores of this business', 'Average Scores of all business'],fontsize = 'x-small')

# generate tfidf
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# remove stopwords
def remove_stopwords(data):
    review = data.apply(lambda x: ' '.join([y for y in x.split() if len(y)>2]))
    review_new = review.apply(lambda x: ' '.join([y for y in x.split() if y not in stop_words]))
    review_new = review_new.apply(str.lower)
    return review_new

review_new = remove_stopwords(review_data['text'])
review_final =  remove_stopwords(review_data['text'][review_data['name']==business_select])
X_review = review_final
y = review_data['sentiment_cag'][review_data['name']==business_select]
tfidf = TfidfVectorizer(ngram_range=(2,3),stop_words = 'english',max_df = 0.30)
X_tfidf = tfidf.fit_transform(X_review)
word = tfidf.get_feature_names_out()


# Train Logistic Model
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, random_state=0)

def text_reg(model,coef_show=1):
    ml = model.fit(X_train, y_train)
    acc = ml.score(X_test, y_test)
    print ('Model Accuracy: {}'.format(acc))
    
    if coef_show == 1: 
        coef = ml.coef_.tolist()[0]
        coeff_df = pd.DataFrame({'Word' : word, 'Coefficient' : coef})
        coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
        return coeff_df
    

coeff_df = pd.DataFrame(text_reg(LogisticRegression()))

positive = coeff_df.head(20).to_string(index = False)
negative = coeff_df.tail(20).to_string(index = False)

tab2_title_1 = '<p style="font-family:sans-serif; color:Black; font-size: 17px;">Sentiment Scores by Different Categories</p>'
#tab2_title_2 = '<p style="font-family:sans-serif; color:Black; font-size: 17px;">Sentiment Scores over Time v.s. Overall Scores</p>'

with tab3:

	with st.container():
		col1, col2 = st.columns(2)
		with col1:
			col1.markdown(tab2_title_1, unsafe_allow_html=True)
			#col1.write("**Sentiment Scores by Different Categories**")
		#with col2:
		#	col2.markdown(tab2_title_2, unsafe_allow_html=True)
		#	#col2.write("**Sentiment Scores over Time v.s. Overall Scores**")

	with st.container():
		st.write("Empty means that this category doesn't contain any reviews.")


	with st.container():
		col1, col2 = st.columns(2)
		with col1:
			col1.pyplot(fig_1)
		with col2:
			col2.pyplot(fig_2)

	
	with st.container():
		col1, col2 = st.columns(2)
		with col1:
			col1.write("**Positive Key Words** :thumbsup:")
		with col2:
			col2.write("**Negative Key Words** :thumbsdown:")

	with st.container():
		col1, col2 = st.columns(2)
		with col1:
			col1.text(positive)
		with col2:
			col2.text(negative)

meat_data = review_segment[(review_segment['text'].str.contains('chicken|fish|chick|pork|beef|noodle|roll|egg|rice|soup|dumpling'))&(review_data['name']==business_select)]
model = LogisticRegression()
tfidf_meat = TfidfVectorizer(ngram_range=(2,2),stop_words = 'english',max_df = 0.7)
X_meat = tfidf_meat.fit_transform(meat_data['text'])
word_meat = pd.Series(tfidf_meat.get_feature_names_out())[pd.Series(tfidf_meat.get_feature_names_out()).str.contains('chicken|fish|chick|pork|beef')]
meat_index = word_meat.index
word = pd.Series(tfidf_meat.get_feature_names_out())
ml_meat = model.fit(X_meat, meat_data['sentiment_cag'])
coef = ml_meat.coef_.tolist()[0]
coef_meat = [coef[i] for i in meat_index]
coeff_df = pd.DataFrame({'Word' : word_meat, 'Coefficient' : coef_meat})
coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
positive = coeff_df.head(10).to_string(index = False)
negative = coeff_df.tail(10).to_string(index = False)

with tab4:
	with st.container():
		col1, col2 = st.columns(2)
		with col1:
			col1.write("**Dishes most liked** :yum:")
		with col2:
			col2.write("**Dishes need to imrove** 	:disappointed:")


	with st.container():
		col1, col2 = st.columns(2)
		with col1:
			col1.text(positive)
		with col2:
			col2.text(negative)

