# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 18:20:48 2026

@author: ahsbd
"""
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import re

# ==========================================
# 1. LOAD ARTIFACTS
# ==========================================
print("Loading model and encoders...")
# Ensure the path matches where your files are stored
try:
    model = joblib.load('lgbm_model.pkl')
    address_encoder = joblib.load('address_encoder.pkl')
    city_encoder = joblib.load('city_encoder.pkl')
    model_columns = joblib.load('model_columns.pkl')
    global_mean_address = joblib.load('global_mean_address.pkl')
    global_mean_city = joblib.load('global_mean_city.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Please ensure 'rental_model_package' folder exists.")
    # Dummy fallback for UI testing if model files are missing
    model, address_encoder, city_encoder = None, None, None

# ==========================================
# 2. DEFINE FEATURE ENGINEERING LOGIC
# ==========================================
premium_areas = [
    'Gulshan', 'Banani', 'Baridhara', 'Uttara Sector 13',
    'Dhanmondi', 'Mirpur DOHS', 'Mohakhali DOHS',
    'Nasirabad Properties', 'Rajuk Uttara Apartment Project'
]

condition_keywords = [
    'Strongly Structured', 'Tastefully Designed', 'Strongly Constructed',
    'Well-Constructed', 'Elegant', 'Spacious', 'Excellent',
    'Marvelous', 'Perfect', 'Smartly Priced'
]

# ==========================================
# 3. SMART PARSING FUNCTION (UPDATED)
# ==========================================
def auto_fill_fields(raw_text):
    """
    Uses Regex to extract Area, Beds, Baths from a pasted description.
    Property Type is hardcoded to 'Flat'.
    """
    text = str(raw_text)
    
    # Extract Area
    area_match = re.search(r'(\d+)\s*(?:sqft|sq\.ft\.|sft|s\.ft|square feet)', text, re.IGNORECASE)
    area = int(area_match.group(1)) if area_match else 1000 
    
    # Extract Beds
    bed_match = re.search(r'(\d+)\s*(?:bed|bedroom|bd)', text, re.IGNORECASE)
    beds = int(bed_match.group(1)) if bed_match else 2
    
    # Extract Baths
    bath_match = re.search(r'(\d+)\s*(?:bath|bathroom|bt)', text, re.IGNORECASE)
    baths = int(bath_match.group(1)) if bath_match else 2
    
    # Use the raw text as the title if it's short, otherwise truncate
    title = text if len(text) < 100 else text[:97] + "..."
    
    # Smart Address Logic
    address = "Dhaka"
    if "," in text:
        address = text.split(",")[-1].strip()
    else:
        for premium_area in premium_areas:
            if premium_area.lower() in text.lower():
                address = premium_area
                break

    
    return title, address, area, beds, baths

# ==========================================
# 4. PREDICTION FUNCTION (UPDATED)
# ==========================================
def predict_price(title, address, area, beds, bath):
    """
    Takes user inputs, processes them to match training format, and predicts price.
    Property Type is forced to 'Flat'.
    """
    # Fallback if model failed to load
    if model is None:
        return "Model not loaded. Check console."

    # A. Create DataFrame for input
    # 'type' is hardcoded to "Flat"
    input_data = pd.DataFrame({
        'title': [title],
        'address': [address],
        'area': [area],
        'beds': [beds],
        'bath': [bath],
        'type': ["Flat"]
    })
    
    # B. Clean Area
    input_data['area'] = pd.to_numeric(input_data['area'], errors='coerce').fillna(0)
    
    # C. Extract Text Features
    input_data['is_premium_area'] = input_data['title'].apply(
        lambda t: int(any(area in t for area in premium_areas))
    )
    input_data['is_high_condition'] = input_data['title'].apply(
        lambda t: int(any(word in t for word in condition_keywords))
    )
    input_data['is_fully_furnished'] = input_data['title'].str.contains(
        'Fully Furnished', case=False, na=False
    ).astype(int)
    input_data['is_luxury'] = input_data['title'].str.contains(
        'Penthouse|Sky View', case=False
    ).astype(int)
    
    # City Extraction
    input_data['city'] = input_data['address'].apply(lambda x: str(x).split(',')[-1].strip())
    
    # D. Math Features
    input_data['baths_per_bed'] = input_data['bath'] / (input_data['beds'] + 1e-6)
    input_data['beds_minus_baths'] = input_data['beds'] - input_data['bath']
    input_data['total_rooms'] = input_data['beds'] + input_data['bath']
    input_data['beds_times_bath'] = input_data['beds'] * input_data['bath']
    
    # E. Log & Interaction Features
    input_data['area_log'] = np.log1p(input_data['area'])
    input_data['area_x_premium'] = input_data['area'] * input_data['is_premium_area']
    input_data['area_x_furnished'] = input_data['area'] * input_data['is_fully_furnished']
    
    # F. Target Encoding
    input_data['address_encoded'] = input_data['address'].map(address_encoder).fillna(global_mean_address)
    input_data['city_encoded'] = input_data['city'].map(city_encoder).fillna(global_mean_city)
    
    # G. One-Hot Encode
    input_data = pd.get_dummies(input_data, columns=['type'], drop_first=True)
    
    # H. Align Columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    
    # I. Predict
    try:
        prediction_log = model.predict(input_data)
        prediction_price = np.expm1(prediction_log)[0]
        
        # MAE Range
        lower_bound = prediction_price - 3283
        upper_bound = prediction_price + 3283
        
        return f"🏠 **Predicted Rent:** {prediction_price:,.0f} BDT\n📉 **Estimated Range:** {lower_bound:,.0f} - {upper_bound:,.0f} BDT"
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# ==========================================
# 5. GRADIO INTERFACE SETUP (UPDATED)
# ==========================================

with gr.Blocks(title="BD Rental Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 Bangladesh Rental Price Predictor (Flat Edition)")
    gr.Markdown("Enter the flat details below to get an estimated rent price.")
    
    with gr.Accordion("📋 Paste Listing Here (Auto-Fill)", open=False):
        gr.Markdown("Copy a property description and paste it below.")
        raw_text_input = gr.Textbox(label="Raw Listing Text", placeholder="e.g. 'Spacious 1200 sqft flat in Gulshan 1, 3 beds, 2 baths...'", lines=3)
        auto_fill_btn = gr.Button("✨ Auto-Fill Form", size="sm")
    
    gr.Markdown("### Property Details")
    
    with gr.Row():
        with gr.Column():
            title_input = gr.Textbox(label="Property Title", lines=2)
            address_input = gr.Textbox(label="Address")
            # --- TYPE INPUT REMOVED ---
        with gr.Column():
            area_input = gr.Number(label="Area (Sq Ft)", value=1200)
            beds_input = gr.Slider(1, 6, step=1, label="Bedrooms", value=2)
            bath_input = gr.Slider(1, 6, step=1, label="Bathrooms", value=2)
    
    predict_btn = gr.Button("Predict Price", variant="primary", size="lg")
    output = gr.Markdown(label="Prediction Result")
    
    # --- EVENTS UPDATED ---
    
    # 1. Auto-fill Logic: Removed type_input from inputs/outputs
    auto_fill_btn.click(
        fn=auto_fill_fields,
        inputs=raw_text_input,
        outputs=[title_input, address_input, area_input, beds_input, bath_input]
    )
    
    # 2. Predict Logic: Removed type_input from inputs
    predict_btn.click(
        fn=predict_price,
        inputs=[title_input, address_input, area_input, beds_input, bath_input],
        outputs=output
    )
    
    # 3. Examples: Removed type_input
    gr.Examples(
        examples=[
            ["Spacious Flat in Gulshan, Fully Furnished", "Gulshan 1, Dhaka", 1800, 3, 3],
            ["Simple flat in Dhanmondi, well constructed", "Dhanmondi 32, Dhaka", 1200, 2, 2]
        ],
        inputs=[title_input, address_input, area_input, beds_input, bath_input]
    )

if __name__ == "__main__":
    demo.launch()