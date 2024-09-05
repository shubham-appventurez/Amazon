import re
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter, DayLocator
import streamlit as st
import psycopg2
from transformers import pipeline, AutoTokenizer
import io

# PostgreSQL database credentials
DB_HOST = 'localhost'
DB_NAME = 'amazon2'
DB_USER = 'amazon2'
DB_PASSWORD = 'amazon2'

# Sentiment Analysis Pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def convert_amazon_to_buyhatke(amazon_url):
    asin_match = re.search(r'/dp/([A-Z0-9]{10})', amazon_url)
    if asin_match:
        asin = asin_match.group(1)
        buyhatke_url = f"https://buyhatke.com/api/productData?pos=6326&pid={asin}"
        return buyhatke_url
    else:
        raise ValueError("Invalid Amazon URL or ASIN not found.")

def get_internal_pid(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        internal_pid = data['data']['internalPid']
        return internal_pid
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except KeyError:
        st.error("Key 'internalPid' not found in the response.")
    except ValueError:
        st.error("Failed to decode JSON response.")

def create_custom_url(internal_pid):
    return f"https://buyhatke.com/amazon-com-price-in-india-6326-{internal_pid}"

def scrape_data_from_custom_url(custom_url):
    try:
        response = requests.get(custom_url)
        response.raise_for_status()
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        script_tag = soup.find('script', string=re.compile('predictedData'))

        if script_tag:
            script_content = script_tag.string
            match = re.search(r'predictedData:(".*?")', script_content)
            if match:
                predicted_data = match.group(1).strip('"')
                cleaned_data = predicted_data.replace('~*~*', '~').replace('~*', '~')
                entries = cleaned_data.split('~')
                dates = []
                prices = []
                i = 0
                while i < len(entries) - 1:
                    date_entry = entries[i].strip()
                    price_entry = entries[i + 1].strip()
                    if re.match(r'\d{4}-\d{2}-\d{2}', date_entry) and re.match(r'\d+\.\d+', price_entry):
                        dates.append(datetime.strptime(date_entry, '%Y-%m-%d %H:%M:%S'))
                        prices.append(float(price_entry))
                    i += 2
                return dates, prices
            else:
                st.error("Predicted data not found")
        else:
            st.error("Script tag with predictedData not found")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

def plot_price_history(dates, prices):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, prices, marker='o', linestyle='-', color='b')
    ax.set_title('Price History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.tight_layout()
    return fig

def scrape_product_details(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "DNT": "1",
        "Connection": "close",
        "Upgrade-Insecure-Requests": "1"
    }

    try:
        payload = {'api_key': 'c25b32150403a78dedf2c425dbf758a2', 'url': url}
        response = requests.get('https://api.scraperapi.com/', params=payload, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html5lib")

        title_elem = soup.find("span", attrs={"id": 'productTitle'})
        title = title_elem.get_text().strip().replace(',', '') if title_elem else "NA"

        price_elem = soup.find("span", class_='a-offscreen')
        price = price_elem.get_text().strip() if price_elem else "NA"
        price = re.sub(r'[^\d.]', '', price) if price != "NA" else "0"
        price = float(price)

        reviews_count_elem = soup.find("span", attrs={'id': 'acrCustomerReviewText'})
        reviews_count = reviews_count_elem.get_text().strip().replace(',', '') if reviews_count_elem else "NA"

        availability_elem = soup.find("div", attrs={'id': 'availability'})
        availability = availability_elem.find("span").get_text().strip().replace(',', '') if availability_elem else "NA"

        description_elem = soup.find("div", attrs={'id': 'productDescription'})
        description = description_elem.find("p").get_text().strip().replace(',', '') if description_elem and description_elem.find("p") else "NA"

        images = [img["src"] for tag in soup.find_all("div", class_="imgTagWrapper") for img in [tag.find("img")] if img and img.has_attr("src")]

        seller_elem = soup.find("a", attrs={'id': 'sellerProfileTriggerId'})
        seller = seller_elem.get_text().strip().replace(',', '') if seller_elem else "Amazon"

        accordion_caption_elem = soup.find("div", class_="a-row accordion-caption")
        used_price_elem = accordion_caption_elem.find_next("span", class_="a-offscreen") if accordion_caption_elem else None
        used_price = used_price_elem.get_text().strip() if used_price_elem else "NA"
        used_price = re.sub(r'[^\d.]', '', used_price) if used_price != "NA" else "0"
        used_price = float(used_price)

        used_condition_elem = soup.find("div", class_="a-row accordion-caption").find_next("span", class_="a-text-bold") if soup.find("div", class_="a-row accordion-caption") else None
        used_condition = used_condition_elem.get_text().strip() if used_condition_elem else "NA"
        used_condition = used_condition.split('-')[1].strip() if used_condition != "NA" and len(used_condition.split('-')) > 1 else "NA"

        discount = round(((price - used_price) / price * 100), 2) if price > 0 and used_price > 0 else 0

        reviews = []
        review_divs = soup.find_all("div", attrs={"data-hook": "review"})
        for review in review_divs:
            review_title = review.find("a", attrs={"data-hook": "review-title"}).get_text().strip() if review.find("a", attrs={"data-hook": "review-title"}) else "NA"
            review_rating = review.find("i", attrs={"data-hook": "review-star-rating"}).get_text().strip() if review.find("i", attrs={"data-hook": "review-star-rating"}) else "NA"
            review_text = review.find("span", attrs={"data-hook": "review-body"}).get_text().strip() if review.find("span", attrs={"data-hook": "review-body"}) else "NA"
            review_author = review.find("span", attrs={"class": "a-profile-name"}).get_text().strip() if review.find("span", attrs={"class": "a-profile-name"}) else "NA"
            review_date = review.find("span", attrs={"data-hook": "review-date"}).get_text().strip() if review.find("span", attrs={"data-hook": "review-date"}) else "NA"
            
            reviews.append({
                "title": review_title,
                "rating": review_rating,
                "text": review_text,
                "author": review_author,
                "date": review_date
            })

        other_sellers = []
        sellers_section = soup.find("div", id="aod-offer-list")

        if sellers_section:
            seller_entries = sellers_section.find_all("div", id="aod-offer")

            for entry in seller_entries:
                seller_name_elem = entry.find("a", class_="a-size-small a-link-normal")
                seller_name = seller_name_elem.get_text().strip() if seller_name_elem else "NA"

                seller_price_elem = entry.find("span", class_="a-price-whole")
                seller_price = seller_price_elem.get_text().strip() if seller_price_elem else "NA"
                seller_price = re.sub(r'[^\d.]', '', seller_price) if seller_price != "NA" else "0"
                seller_price = float(seller_price)

                other_sellers.append({
                    "seller_name": seller_name,
                    "price": seller_price
                })

        return {
            'title': title,
            'price': price,
            'reviews_count': reviews_count,
            'availability': availability,
            'description': description,
            'discount': discount,
            'used_price': used_price,
            'used_condition': used_condition,
            'images': images,
            'seller': seller,
            'reviews': reviews,
            'other_sellers': other_sellers
        }

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")

def predict_return_rate(reviews):
    max_length = 512  # Maximum length for the model

    truncated_reviews = []
    for review in reviews:
        review_text = review["text"]
        # Tokenize and truncate the review text
        tokens = tokenizer.encode(review_text, max_length=max_length, truncation=True)
        truncated_review = tokenizer.decode(tokens, skip_special_tokens=True)
        truncated_reviews.append(truncated_review)  

    sentiments = sentiment_pipeline(truncated_reviews)
    positive_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'POSITIVE')
    return_rate = (len(truncated_reviews) - positive_count) / len(truncated_reviews) * 100 if truncated_reviews else 0
    return round(return_rate, 2)

def save_to_postgresql(data):
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO product_details (title, price, reviews_count, availability, description, discount, used_price, used_condition)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (data['title'], data['price'], data['reviews_count'], data['availability'],
              data['description'], data['discount'], data['used_price'], data['used_condition']))

        conn.commit()

        for img_url in data['images']:
            cursor.execute("""
            INSERT INTO product_images (product_title, image_url)
            VALUES (%s, %s)
            """, (data['title'], img_url))

        for review in data['reviews']:
            cursor.execute("""
            INSERT INTO product_reviews (product_title, review_title, review_rating, review_text, review_author, review_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            """, (data['title'], review['title'], review['rating'], review['text'],
                  review['author'], review['date']))

        for seller in data['other_sellers']:
            cursor.execute("""
            INSERT INTO other_sellers (product_title, seller_name, price)
            VALUES (%s, %s, %s)
            """, (data['title'], seller['seller_name'], seller['price']))

        conn.commit()
        cursor.close()
        conn.close()

    except psycopg2.Error as e:
        st.error(f"Database error: {e}")

def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Select a Page', ['Home', 'Scrape Product'])

    if page == 'Home':
        st.title('Welcome to the Amazon Scraper App')
        st.write("This app allows you to scrape product details from Amazon and visualize price history from Buyhatke.")

    elif page == 'Scrape Product':
        st.title('Scrape Amazon Product Details')

        amazon_url = st.text_input("Enter Amazon Product URL")
        if st.button('Scrape'):
            if amazon_url:
                try:
                    buyhatke_url = convert_amazon_to_buyhatke(amazon_url)
                    internal_pid = get_internal_pid(buyhatke_url)
                    custom_url = create_custom_url(internal_pid)
                    dates, prices = scrape_data_from_custom_url(custom_url)

                    if dates and prices:
                        fig = plot_price_history(dates, prices)
                        st.pyplot(fig)

                    data = scrape_product_details(amazon_url)
                    if data:
                        save_to_postgresql(data)
                        st.success("Product details saved to database.")

                        st.subheader("Product Details")
                        st.write(f"**Title:** {data['title']}")
                        st.write(f"**Price:** {data['price']}")
                        st.write(f"**Reviews Count:** {data['reviews_count']}")
                        st.write(f"**Availability:** {data['availability']}")
                        st.write(f"**Description:** {data['description']}")
                        st.write(f"**Discount:** {data['discount']}%")
                        st.write(f"**Used Price:** {data['used_price']}")
                        st.write(f"**Used Condition:** {data['used_condition']}")
                        st.write("**Images:**")
                        for img_url in data['images']:
                            st.image(img_url, use_column_width=True)

                        # Calculate return rate and display
                        return_rate = predict_return_rate(data['reviews'])
                        st.write(f"**Predicted Return Rate:** {return_rate:.2f}%")
                        st.write(f"**Predicted Return Rate (Debug):** {return_rate:.2f}%")  # Debugging line

                        st.subheader("Reviews")
                        for review in data['reviews']:
                            st.write(f"**{review['title']}**")
                            st.write(f"Rating: {review['rating']}")
                            st.write(f"Date: {review['date']}")
                            st.write(f"Author: {review['author']}")
                            st.write(f"Review: {review['text']}")
                            st.write("---")

                        st.subheader("Other Sellers")
                        for seller in data['other_sellers']:
                            st.write(f"**Seller:** {seller['seller_name']}")
                            st.write(f"Price: {seller['price']}")
                            st.write("---")
                except ValueError as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
