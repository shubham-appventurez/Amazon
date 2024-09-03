import streamlit as st
import requests
from bs4 import BeautifulSoup
import psycopg2
from transformers import pipeline, AutoTokenizer
import re
import time

# PostgreSQL database credentials
DB_HOST = 'localhost'
DB_NAME = 'amazon2'
DB_USER = 'amazon2'
DB_PASSWORD = 'amazon2'

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Function to scrape product details from Amazon URL
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
        response.raise_for_status()  # Raise HTTPError for bad responses

        soup = BeautifulSoup(response.content, "html5lib")

        # Extracting product title
        title_elem = soup.find("span", attrs={"id": 'productTitle'})
        title = title_elem.get_text().strip().replace(',', '') if title_elem else "NA"

        # Extracting product price
        price_elem = soup.find("span", class_='a-offscreen')
        price = price_elem.get_text().strip() if price_elem else "NA"

        # Use regex to remove commas and the sign
        if price != "NA":
            price = re.sub(r'[^\d.]', '', price)
            price = float(price)
        else:
            price = 0

        # Extracting review count
        reviews_count_elem = soup.find("span", attrs={'id': 'acrCustomerReviewText'})
        reviews_count = reviews_count_elem.get_text().strip().replace(',', '') if reviews_count_elem else "NA"

        # Extracting availability status
        availability_elem = soup.find("div", attrs={'id': 'availability'})
        availability = availability_elem.find("span").get_text().strip().replace(',', '') if availability_elem else "NA"

        # Extracting product description
        description_elem = soup.find("div", attrs={'id': 'productDescription'})
        description = description_elem.find("p").get_text().strip().replace(',', '') if description_elem and description_elem.find("p") else "NA"

        # Extracting product images
        images = []
        img_tags = soup.find_all("div", class_="imgTagWrapper")
        for tag in img_tags:
            img = tag.find("img")
            if img and img.has_attr("src"):
                images.append(img["src"])

        # Extracting seller details
        seller_elem = soup.find("a", attrs={'id': 'sellerProfileTriggerId'})
        seller = seller_elem.get_text().strip().replace(',', '') if seller_elem else "Amazon"   
        

        # Extracting used product details
        accordion_caption_elem = soup.find("div", class_="a-row accordion-caption")
        used_price_elem = accordion_caption_elem.find_next("span", class_="a-offscreen") if accordion_caption_elem else None
        used_price = used_price_elem.get_text().strip() if used_price_elem else "NA"

        # Use regex to remove commas and the dollar sign
        if used_price != "NA":
            used_price = re.sub(r'[^\d.]', '', used_price)
            used_price = float(used_price)
        else:
            used_price = 0
        
        used_condition_elem = soup.find("div", class_="a-row accordion-caption").find_next("span", class_="a-text-bold") if soup.find("div", class_="a-row accordion-caption") else None
        used_condition = used_condition_elem.get_text().strip() if used_condition_elem else "NA"

        if used_condition != "NA":
            parts = used_condition.split('-')
            if len(parts) > 1:
                used_condition = parts[1].strip()

        # Calculate discount
        discount = round(((price - used_price) / price * 100), 2) if price > 0 and used_price > 0 else 0

        # Extracting reviews
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


             # Extracting other sellers
        other_sellers = []
        sellers_section = soup.find("div", id="aod-offer-list")  # This may vary; adjust based on actual HTML structure

        if sellers_section:
            seller_entries = sellers_section.find_all("div", id="aod-offer")

            for entry in seller_entries:
                seller_name_elem = entry.find("a", class_="a-size-small a-link-normal")
                seller_name = seller_name_elem.get_text().strip() if seller_name_elem else "Unknown"

                seller_price_elem = entry.find("span", class_="a-offscreen")
                seller_price = seller_price_elem.get_text().strip().replace(',', '') if seller_price_elem else "NA"

                seller_condition_elem = entry.find("div", id="aod-offer-heading")
                seller_condition = seller_condition_elem.get_text().strip() if seller_condition_elem else "NA"

                seller_shipping_elem = entry.find("span", class_="a-size-small a-color-base")
                seller_shipping = seller_shipping_elem.get_text().strip().replace(',', '') if seller_shipping_elem else "Free"

                other_sellers.append({
                    "name": seller_name,
                    "price": seller_price,
                    "condition": seller_condition,
                    "shipping": seller_shipping
                })

        return {
            "title": title,
            "price": price,
            "reviews_count": reviews_count,
            "availability": availability,
            "description": description,
            "images": images,
            "seller": seller,
            "used_price": used_price,
            "used_condition": used_condition,
            "discount": discount,
            "reviews": reviews,
            "other_sellers": other_sellers
        }

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching {url}: {e}")
        return None

# Function to predict return rate based on reviews
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

# Function to save product details to PostgreSQL database
def save_to_database(data):
    # Convert the list of images into a PostgreSQL array literal format
    images_array = '{' + ','.join(f'"{img}"' for img in data['images']) + '}'

    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO products (title, price, reviews_count, availability, description, images, seller, used_price, used_condition, discount)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (data['title'], data['price'], data['reviews_count'], data['availability'], data['description'], images_array, data['seller'], data['used_price'], data['used_condition'], data['discount']))
    conn.commit()
    cursor.close()
    conn.close()


def product_exists(title):
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute("SELECT price FROM products WHERE title = %s", (title,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result

# Streamlit app with multiple pages
st.set_page_config(page_title="Amazon Product Scraper", layout="wide")

# Load custom CSS
with open('style.css') as f:    
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Scrape Product"])

if page == "Home":
    st.title("Welcome to the Amazon Product Scraper!")
    st.write("Use this app to scrape product details from Amazon, analyze reviews, and save data to a database.")

elif page == "Scrape Product":
    st.title("Scrape Amazon Product Details")

    url = st.text_input("Enter Amazon Product URL")
    if st.button("Scrape Product Details"):
        if url:
            loading_placeholder = st.empty()

            # Display the loading image
            loading_html = """
                    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; height: 50vh;">
                        <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExMGJoOTNkOGF2NzRweHlsNnIxZWJ5dXRhYW8xbWQ3bGdjOXdmbXp1YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l0Iy6QHeT2PENtdEQ/giphy.webp" alt="Loading..." style="width: 200px; height: 200px;" />
                        <p style="font-size: 20px; margin-top: 10px;">Scraping product details...</p>
                    </div>
                    """
                    # Use markdown to render HTML
            loading_placeholder.markdown(loading_html, unsafe_allow_html=True)           
 
            data = scrape_product_details(url)
            loading_placeholder.empty()  # Clear the loading image
            if data:
                # Check if a product with the same title already exists in the database
                existing_product = product_exists(data['title'])

                if existing_product:
                    st.write("### Product Already Exists in Database")
                    previous_price_str = existing_product[0]  # Get the price as a string
                    try:
                        previous_price = float(previous_price_str)  # Convert to float
                        st.write(f"**Previous Price:** {previous_price:.2f}")
                    except ValueError:
                        st.write("**Previous Price:** Data error")


                # Save the new product details
                save_to_database(data)

                st.success("Product details scraped and saved successfully!")

                # Layout: Images on the top left and details on the right
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.write("### Product Images")
                    for image_url in data['images']:
                        st.image(image_url, use_column_width=True)

                with col2:
                    st.write("### Product Title")
                    st.write(data['title'])
                    st.write("### Price")
                    st.write(f"{data['price']:.2f}")
                    st.write("### Used Price")
                    st.write(f"{data['used_price']:.2f}")
                    st.write("### Used Condition")
                    st.write(data['used_condition'])
                    st.write("### Discount")
                    st.write(f"{data['discount']}%")
                    st.write("### Reviews Count")
                    st.write(data['reviews_count'])
                    st.write("### Availability")
                    st.write(data['availability'])
                    st.write("### Description")
                    st.write(data['description'])
                    st.write("### Seller")
                    st.write(data['seller'])

                    if data['other_sellers']:
                        st.write("### Other Sellers")
                        for seller in data['other_sellers']:
                            st.write(f"**Seller:** {seller['name']}")
                            st.write(f"**Price:** {seller['price']}")
                            st.write(f"**Condition:** {seller['condition']}")
                            st.write(f"**Shipping:** {seller['shipping']}")
                            st.write("---")
                    
                    # Add return rate bar
                    st.write("### Return Rate")
                    return_rate = predict_return_rate(data['reviews'])
                    st.write(f"{return_rate}%")
                    st.progress(return_rate/100)
                    st.write("### Reviews")
                    for review in data['reviews']:
                        st.write(f"**{review['title']}** ({review['rating']})")
                        st.write(f"By {review['author']} on {review['date']}")
                        st.write(review['text'])
                        st.write("---")
            else:
                st.error("Error fetching product details. Please try again later.")
    else:
        st.warning("Please enter a valid Amazon product URL.")
