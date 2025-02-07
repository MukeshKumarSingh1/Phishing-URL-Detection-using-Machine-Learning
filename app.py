import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
import re
import urllib.parse
from tld import get_tld

# Define feature extraction (as previously discussed)
def extract_url_features(url):
    # URL Parsing
    parsed_url = urllib.parse.urlparse(url)
    
    # 1. URL Length
    url_length = len(url)
    
    # 2. Number of Subdomains
    subdomain_count = len(parsed_url.hostname.split('.')) - 2  # Subdomains before the domain
    
    # 3. Check if HTTPS is used
    is_https = 1 if parsed_url.scheme == 'https' else 0
    
    # 4. Number of Special Characters
    special_chars_count = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url))
    
    # 5. Path Length
    path_length = len(parsed_url.path)
    
    # 6. Number of Query Parameters
    query_params = urllib.parse.parse_qs(parsed_url.query)
    num_query_params = len(query_params)
    
    # 7. Top-Level Domain (TLD)
    try:
        tld = get_tld(url)
        tld_length = len(tld)
    except:
        tld_length = 0  # In case TLD extraction fails

    # 8. Domain Age (Placeholder, as we need to query domain registration info)
    domain_age = 0  # For simplicity, we assume domain age as 0 (can be enhanced with external APIs)

    # 9. URL Entropy
    url_entropy = -sum([(url.count(c) / len(url)) * (url.count(c) / len(url)) for c in set(url)])

    # 10. Path to Domain Ratio
    domain_length = len(parsed_url.hostname) if parsed_url.hostname else 0
    path_to_domain_ratio = path_length / domain_length if domain_length > 0 else 0
    
    return np.array([
        url_length,
        subdomain_count,
        is_https,
        special_chars_count,
        path_length,
        num_query_params,
        tld_length,
        domain_age,
        url_entropy,
        path_to_domain_ratio
    ])

# Load the pre-trained model (using the model you trained earlier)
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load model weights (you should save the trained weights and load them here)
model.load_weights('model_weights.h5')

# Create Streamlit App
st.title('Phishing URL Detection')
st.write('Enter a URL below to check if it is phishing or not.')

# Input URL
url_input = st.text_input("Enter URL:")

if url_input:
    features = extract_url_features(url_input)
    
    # Reshape for prediction
    features = features.reshape(1, -1)
    
    # Normalize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Prediction
    prediction = model.predict(features)
    
    # Show result
    if prediction[0][0] > 0.5:
        st.error("This URL is a Phishing URL!")
    else:
        st.success("This URL is Safe!")
