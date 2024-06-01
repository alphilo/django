from django.shortcuts import render

# Create your views here.
# from .models import DataPoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
from tqdm import tqdm 
import re 
import nltk 
nltk.download('punkt') 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer 
from wordcloud import WordCloud
from tqdm import tqdm 
import re 
import nltk 
nltk.download('punkt') 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer 
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer 
# from sklearn.feature_extraction.text import CountVectorizer 


def preprocess_text(text_data): 
	preprocessed_text = [] 
	
	for sentence in tqdm(text_data): 
		sentence = re.sub(r'[^\w\s]', '', sentence) 
		preprocessed_text.append(' '.join(token.lower() 
								for token in str(sentence).split() 
								if token not in stopwords.words('english'))) 

	return preprocessed_text

# def preprocess_text(text_data): 
# 	preprocessed_text = [] 
	
# 	for sentence in tqdm(text_data): 
# 		sentence = re.sub(r'[^\w\s]', '', sentence) 
# 		preprocessed_text.append(' '.join(token.lower() 
# 								for token in str(sentence).split() 
# 								if token not in stopwords.words('english'))) 

# 	return preprocessed_text

def fake_news(request):
    data = pd.read_csv('data/news2.csv',index_col=0) 
    data.head()
    data.shape
    # print("Data shape:", data.shape);
    data = data.drop(["title", "subject","date"], axis = 1)
    data.isnull().sum()
    # print("Data size:", data.isnull().sum());
    # Shuffling 
    data = data.sample(frac=1) 
    data.reset_index(inplace=True) 
    data.drop(["index"], axis=1, inplace=True) 
    sns_plot = sns.countplot(data=data, 
                x='class', 
                order=data['class'].value_counts().index)
       
    # Save the plot to a temporary file
    plot_path = "data_visualization/static/sns_plot.png"
    sns_plot.figure.savefig(plot_path)
    plt.close()
    plot_path = '../../static/sns_plot.png'

    preprocessed_review = preprocess_text(data['text'].values) 
    data['text'] = preprocessed_review
    # print("Data:", preprocessed_review);
    # Real 
    consolidated = ' '.join( 
        word for word in data['text'][data['class'] == 1].astype(str)) 
    wordCloud = WordCloud(width=1600, 
                        height=800, 
                        random_state=21, 
                        max_font_size=110, 
                        collocations=False) 
    # print("Word cloud:", wordCloud);
    plt.figure(figsize=(15, 10)) 
    plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear') 
    plt.axis('off') 
    # plt.show()
    real_plot_path = 'data_visualization/static/real_plot_path.png' 
    plt.savefig(real_plot_path) 
    real_plot_path = '../../static/real_plot_path.png'
	 
    # Fake 
    consolidated = ' '.join( 
        word for word in data['text'][data['class'] == 0].astype(str)) 
    wordCloud = WordCloud(width=1600, 
                        height=800, 
                        random_state=21, 
                        max_font_size=110, 
                        collocations=False) 
    plt.figure(figsize=(15, 10)) 
    plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear') 
    plt.axis('off') 
    # plt.show() 

    fake_plot_path = 'data_visualization/static/fake_plot_path.png' 
    plt.savefig(fake_plot_path) 
    fake_plot_path = '../../static/fake_plot_path.png'

    # x_values = [point.x_value for point in data]
    # y_values = [point.y_value for point in data]

    # plt.scatter(x_values, y_values)
    # plt.xlabel('X Values')
    # plt.ylabel('Y Values')
    # plt.title('Data Visualization')
    # plt.grid(True)

    # # Save plot to a temporary file
    # plot_path = 'data_visualization/static/plot.png'
    # plt.savefig(plot_path)
    # plot_path = '../../static/plot.png';
    print("Fake plot path", fake_plot_path)
    return render(request, 'data_visualization/fake_news.html', {'plot_path': plot_path, 'real_plot_path': real_plot_path, 'fake_plot_path': fake_plot_path})