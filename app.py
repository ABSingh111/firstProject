import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from io import BytesIO
import requests
from PIL import Image
import re
import os
from streamlit_lottie import st_lottie



# Function to load Lottie animations
def load_lottieurl(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Load animations
lottie_shopping = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_tfb3estd.json")
lottie_explore = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_z4cshyhf.json")

# Set page configuration
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛒",
    layout="wide",
)

# Apply background style
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to bottom, #e3f2fd, #ffffff);
        color: #333333;
    }
    .title {
        text-align: center;
        font-size: 2rem;
        color: #4A90E2;
    }
    .subtitle {
        text-align: center;
        font-size: 1.5rem;
        color: #5a5a5a;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None



# Load ResNet50 model
resnet_model = ResNet50(weights='imagenet', include_top=False)

# Load pre-trained CNN + RNN model
model = load_model('models/my_model.h5')

# Load dataset
if not os.path.exists('data/amazon.csv'):
    st.error("Dataset file not found.")
else:
    df = pd.read_csv('data/amazon.csv')

    # Clean price and rating columns
    def clean_column(column):
        return column.apply(lambda x: float(re.sub(r'[^\d.]', '', str(x))) if pd.notnull(x) else np.nan)

    df['discounted_price'] = clean_column(df['discounted_price'])
    df['actual_price'] = clean_column(df['actual_price'])
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Drop rows with NaN values
    df.dropna(subset=['discounted_price', 'actual_price', 'rating'], inplace=True)

    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['product_name'])

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['product_name'])

    # Linear Regression model
    features = df[['discounted_price', 'actual_price', 'rating']]
    linear_model = LinearRegression()
    linear_model.fit(features, df['rating'])

# Function to preprocess images for ResNet50
def process_image(img_url):
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        img_features = resnet_model.predict(img_array)
        return img_features
    except (requests.RequestException, Image.UnidentifiedImageError):
        return None

# Get popular products based on rating
def get_popular_products(top_n=5):
    return df.nlargest(top_n, 'rating')

# Recommendation function
def hybrid_recommendations(search_query=None, top_n=5):
    if not search_query:
        st.write("Showing popular items:")
        popular_products = get_popular_products(top_n)
        for _, product in popular_products.iterrows():
            display_product(product)
    else:
        query_vector = tfidf.transform([search_query])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]

        if not any(cosine_similarities > 0):
            st.warning("The searched product is not available right now. Please try searching for something else. Thank you!")
            st.write("Here are some popular items you may like:")
            popular_products = get_popular_products(top_n)
            for _, product in popular_products.iterrows():
                display_product(product)
            return

        for idx in top_indices:
            product = df.iloc[idx]
            display_product(product)

# Function to display a product
def display_product(product):
    img_url = product.get('img_link', None)
    if img_url:
        img_array = process_image(img_url)
        if img_array is None:
            return
        st.image(img_url, use_column_width=True)
    st.write(product.get('product_name', 'No Name Available'))
    st.write(f"Discounted Price: ₹{product.get('discounted_price', 'N/A')}")
    st.write(f"Actual Price: ₹{product.get('actual_price', 'N/A')}")
    st.write(f"Rating: {product.get('rating', 'N/A')}")

# Home page
# Home page content
st.markdown("<div class='title'>🌟 Welcome to Product Recommendation System 🌟</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your personalized shopping assistant, just a click away!</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #4A90E2;'> Product Recommendation System</h1>", unsafe_allow_html=True)
st_lottie(lottie_shopping, height=300, key="shopping_animation")


# Call-to-action section
st.write("## Explore our features:")
st.markdown(
    """
    - 🔍 **Search for Products**: Find exactly what you're looking for.
    - 🌟 **Top Recommendations**: See what’s trending and highly rated.
    - 🛒 **Admin Features**: Add or remove products easily.
    """
)

# Explore button with animation
st.write("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("Start Exploring Products 🚀"):
        st.write("Redirecting to product recommendations...")  # Replace with actual navigation if applicable
        st_lottie(lottie_explore, height=300, key="explore_animation")

# Sidebar for account login/logout
account_type = st.sidebar.radio("Select Account Type:", ["User", "Admin"])
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.sidebar.success("Logged out successfully!")
else:
    user_id = st.sidebar.text_input("User ID")
    if st.sidebar.button("Login"):
        if user_id:
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.sidebar.success(f"Welcome back, {user_id}!")
        else:
            st.sidebar.error("Please enter a User ID.")

# Admin section
if account_type == "Admin" and st.session_state.logged_in:
    st.sidebar.write("Admin Section")
    # Add product form
    with st.form("Add Product"):
        name = st.text_input("Product Name")
        actual_price = st.number_input("Actual Price", min_value=0.0)
        discounted_price = st.number_input("Discounted Price", min_value=0.0)
        rating = st.number_input("Rating", min_value=0.0, max_value=5.0)
        img_link = st.text_input("Image Link")
        description = st.text_area("Product Description")
        submitted = st.form_submit_button("Add Product")
        if submitted:
            new_product = {
                'product_name': name,
                'actual_price': actual_price,
                'discounted_price': discounted_price,
                'rating': rating,
                'img_link': img_link,
                'about_product': description
            }
            df = pd.concat([df, pd.DataFrame([new_product])], ignore_index=True)
            df.to_csv('data/amazon.csv', index=False)
            st.success("Product added successfully!")
    # Delete product
    st.write("Delete a Product")
    product_name_to_delete = st.selectbox("Select a Product to Delete", df['product_name'])
    if st.button("Delete Product"):
        df = df[df['product_name'] != product_name_to_delete]
        df.to_csv('data/amazon.csv', index=False)
        st.success("Product deleted successfully!")

# Search and recommendations
if st.session_state.logged_in and st.session_state.user_id:
    search_query = st.text_input("🔍 Search for products", placeholder="Type a product name here...")
    hybrid_recommendations(search_query=search_query)
else:
    st.write("Please log in to view personalized recommendations.")
    
import re  # For cleaning price data

def advanced_bargaining_chatbot_with_profit(product_name):
    st.write("🤖 **Welcome to the Smart Bargaining Chatbot!**")

    # Find the product
    product = df[df['product_name'].str.contains(product_name, case=False, na=False)]
    if product.empty:
        st.warning("Product not found. Please try with a valid product name.")
        return

    product = product.iloc[0]

    # Preprocess the prices to clean unwanted symbols and convert to integers
    def clean_price(price):
        """Removes non-numeric symbols and converts to integer."""
        if isinstance(price, str):
            price = re.sub(r'[^\d]', '', price)  # Remove all non-numeric characters
        return int(price) if price else 0

    discounted_price = clean_price(product['discounted_price'])
    actual_price = clean_price(product['actual_price'])

    cost_price = discounted_price * 0.7  # Assume the cost price is 70% of discounted price
    min_profit_margin = 0.1  # Minimum profit margin (10%)
    min_price = cost_price * (1 + min_profit_margin)  # Minimum selling price with profit

    # Initialize session state for negotiation
    if "bargain_state" not in st.session_state:
        st.session_state.bargain_state = {
            "turns": 0,
            "chatbot_price": discounted_price,
            "min_price": min_price,
            "negotiation_complete": False,
            "user_price": None,
        }

    state = st.session_state.bargain_state

    # Display product details
    st.write(f"**Product Name:** {product_name}")
    st.write(f"**Actual Price:** ₹{actual_price}")
    st.write(f"**Discounted Price:** ₹{discounted_price}")
   # st.write(f"🔒 **Minimum Selling Price (hidden):** ₹{round(min_price, 2)}")  # Debugging only
    img_url = product.get('img_link', None)
    if img_url:
        st.image(img_url, use_column_width=True)

    # Bargaining chat logic
    if not state["negotiation_complete"]:
        st.write(f"💬 **Chatbot**: The actual price of **{product_name}** is ₹{actual_price}, and the discounted price is ₹{discounted_price}. Let’s negotiate!")

        # Input for user price
        user_price = st.number_input("💬 Your Offer Price", min_value=0.0, step=100.0, key="user_offer_price")

        # Negotiate button
        negotiate_clicked = st.button("Negotiate")

        # Only process negotiation if the button is clicked
        if negotiate_clicked:
            if user_price == 0:
                st.warning("Please enter your offer price before negotiating!")
                return

            state["turns"] += 1
            state["user_price"] = user_price  # Update user price in session state

            # Handle user price and chatbot response
            if state["user_price"] >= state["chatbot_price"]:
                st.success(f"🤖 **Chatbot**: Deal accepted! 🎉 You can buy it for ₹{state['user_price']}.")
                state["negotiation_complete"] = True
            elif state["user_price"] >= state["min_price"]:
                state["chatbot_price"] -= 0.03 * state["chatbot_price"]  # Reduce price slightly
                st.info(f"🤖 **Chatbot**: Hmm, that’s a bit low. How about ₹{round(state['chatbot_price'], 2)}?")
            else:
                st.warning(f"🤖 **Chatbot**: Sorry, I can’t go below ₹{round(state['min_price'], 2)}.")
            
            # Check for maximum turns
            if state["turns"] >= 3:
                st.warning("🤖 **Chatbot**: Negotiation over! Maximum attempts reached.")
                st.write(f"🔒 **Final Offer:** ₹{round(max(state['chatbot_price'], state['min_price']), 2)}")
                state["negotiation_complete"] = True
    else:
        st.write("💬 **Chatbot**: Negotiation complete. Thank you for using our service!")
        # Reset button to allow a new negotiation
        if st.button("Start New Negotiation"):
            del st.session_state.bargain_state

# Adding the advanced chatbot section
st.write("---")
st.write("## 🤝 Smart AI Bargaining Chatbot")
bargain_product = st.selectbox("Enter a product name to start bargaining", df['product_name'])
start_bargain = st.button("Start Bargaining")

# Only start chatbot if the user initiates it
if start_bargain or "bargain_state" in st.session_state:
    if bargain_product or "bargain_state" in st.session_state:
        advanced_bargaining_chatbot_with_profit(bargain_product)
