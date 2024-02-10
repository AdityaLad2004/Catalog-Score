from fastapi import FastAPI,File
from papermill import execute_notebook
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
@app.post("/run_ipynb")
async def run_ipynb(url: str):
    print(main(url))
    return url

# if __name__ == "__main__":
#     uvicorn main:app --reload

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

def get_product_info(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36'
    }

    webpage = requests.get(url, headers=headers).text
    soup = BeautifulSoup(webpage, 'lxml')

    name = str(soup.find_all('h1')[0].text)
    
    bankoffers = soup.find_all('div', class_='XUp0WS')[0].text.split('T&C')
    bankoffers_count = len(bankoffers)
    
    description = soup.find_all('div', class_='_1mXcCf RmoJUa')[0].text
    description_length = len(description)
    
    no_of_images = len(soup.find_all('img', class_='q6DClP'))
    
    price_str = soup.find_all('div', class_='_30jeq3 _16Jk6d')[0].text
    price = int(price_str.replace('₹', '').replace(',', ''))
    
    ratings = soup.find_all('span', class_='_2_R_DZ')[1].text

    html_string = '<span style="user-select: auto;">&nbsp;1,562 Reviews</span>'
    souprev = BeautifulSoup(html_string, 'html.parser')
    reviews_number = int(souprev.get_text().split()[0].replace(',', ''))

    data = {
        'Name': [name],
        'Bank Offers Count': [bankoffers_count],
        'Description Length': [description_length],
        'No. of Images': [no_of_images],
        'Price': [price],
        'Ratings': [ratings],
        'Reviews Number': [reviews_number]
    }

    df = pd.DataFrame(data)
    df['description'] = soup.find_all('div', class_='_1mXcCf RmoJUa')[0].text
    df['description'] = df['description'].iloc[0] 

    return df,soup

def analyze_sentiment(description):
    sid = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(description)

    total_neg = 0
    total_neu = 0
    total_pos = 0
    total_compound = 0

    for sentence in sentences:
        scores = sid.polarity_scores(sentence)
        total_neg += scores['neg']
        total_neu += scores['neu']
        total_pos += scores['pos']
        total_compound += scores['compound']

    num_sentences = len(sentences)
    avg_neg = total_neg / num_sentences
    avg_neu = total_neu / num_sentences
    avg_pos = total_pos / num_sentences

    overall_score = (avg_pos - avg_neg) * avg_neu
    normalized_score = ((overall_score + 1) / 2) * 9 + 1

    return normalized_score

def calculate_rating(price, reviews, total_purchases):
    mask_cheap = price < 300
    mask_mid = (price >= 300) & (price <= 2000)
    
    review_rate = pd.Series(0, index=price.index)
    review_rate[mask_cheap] = 5 / 100
    review_rate[mask_mid] = 15 / 100
    review_rate[~(mask_cheap | mask_mid)] = 25 / 100
    
    expected_reviews = review_rate * total_purchases
    rating = reviews / expected_reviews * 10
    
    rating.replace([np.inf, -np.inf], np.nan, inplace=True)
    rating.fillna(10, inplace=True)
    
    return rating

def calculate_final_score(product_info,soup):
    price = product_info['Price']
    reviews = product_info['Reviews Number']
    ratings = product_info['Ratings'].iloc[0]
    ratings_cleaned = int(''.join(filter(str.isdigit, ratings)))
    total_purchases = ratings_cleaned

    rating = calculate_rating(price, reviews, total_purchases)
    image_score = product_info['No. of Images'] / 10
    bank_score = product_info['Bank Offers Count'] / 7
    rating_score = rating / 5

    description_score = analyze_sentiment(product_info['description'].iloc[0]) / 10
    
    k = soup.find_all('div', class_='_220jKJ FEJ_PY')[0].text
    if k == "BESTSELLER":
        bestsellerscore = 1
    else:
        bestsellerscore = 0.5

    weight_bank_score = 0.2
    weight_image_score = 0.4
    weight_rating_score = 0.15
    weight_description_score = 0.125
    weight_bs_score = 0.125

    final_score = (bank_score * weight_bank_score) + (image_score * weight_image_score) + (rating_score * weight_rating_score) + (description_score * weight_description_score) + (bestsellerscore * weight_bs_score)

    product_info['Final Score'] = final_score

    return product_info, final_score

# Example usage:
def main(url):
    # url = 'https://www.flipkart.com/vivo-v29e-5g-artistic-red-128-gb/p/itmaaa634624a459?pid=MOBGSZM9BGT9QE9U'
    product_info,soup = get_product_info(url)
    final_product_info, final_score = calculate_final_score(product_info,soup)
    # print("Final Product Information:")
    # print(final_product_info)
    # print("Final Score:", final_score)
    return final_score