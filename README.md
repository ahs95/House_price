# 🏠 Bangladesh Rental Price Predictor

A Machine Learning-powered tool to estimate monthly rental prices for flats in Bangladesh. This project uses **LightGBM** and **Target Encoding** to provide accurate rent predictions based on location, size, and property description.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Key Features

*   **Smart Auto-Fill:** Simply copy-paste a raw property description (e.g., from Facebook Marketplace or Bikroy), and the AI will automatically extract the Area, Beds, Baths, and Address.
*   **Mass Market Focus:** Specifically tuned for standard residential **Flats**, filtering out extreme luxury outliers (Penthouses) and commercial properties.
*   **Location Intelligence:** Uses Target Encoding on both specific `Address` and broader `City` levels to capture hyper-local market trends.
*   **Interactive UI:** Built with **Gradio** for a clean, user-friendly web interface.

## 🎥 Demo
See the tool in action! The short video below demonstrates the user interface and the **Smart Auto-Fill** feature in real-time.

[Watch Demo](https://cap.so/s/jnz2mc43dhsb66x)

## 📊 Model Performance

The model was trained on a dataset of ~7,000 residential listings.

| Metric | Score |
| :--- | :--- |
| **Algorithm** | LightGBM |
| **Validation RMSE** | ~5,807 BDT (24.90%) |
| **Mean Absolute Error (MAE)** | ~3,000 BDT |
| **Primary Drivers** | Address, Area, Bath-to-Bed Ratio |

> *Note: Predictions are intended as market estimates. Actual prices may vary based on negotiation, furnishing quality, and specific building amenities.*

## 🚀 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ahs95/House_price
    cd House_price
    ```

2.  **Install Dependencies**
    Ensure you have Python 3.8+ installed, then install the required packages:
    ```bash
    pip install gradio lightgbm pandas numpy scikit-learn joblib
    ```

## 🖥️ Usage

Once the dependencies are installed, you can launch the application directly.

1.  **Run the App**
    ```bash
    cd rental_model_package
    python app.py
    ```

2.  **Open the Interface**
    The terminal will output a local URL (usually `http://127.0.0.1:7860`). Click this link to open the predictor in your browser.

## 💡 How to Use the Tool

1.  **Manual Entry:** Enter the flat details (Address, Area, Bedrooms, Bathrooms) directly into the form.
2.  **Auto-Fill (Recommended):**
    *   Copy a text listing (e.g., *"Spacious 1200 sqft flat in Gulshan 1, 3 beds, 2 baths, fully furnished"*).
    *   Paste it into the "Paste Listing Here" box.
    *   Click **✨ Auto-Fill Form** to automatically populate the fields.

## 🧠 Project Structure
*   `rental_model_package/`: The directory contains the saved LightGBM model, target encoders, column references, and the main script that implements the Gradio interface and prediction logic.
*   `House_price_model.ipynb`: Jupyter Notebook containing the full data cleaning, and model training pipeline.
*   `Outliers_detection.ipynb`: Jupyter Notebook focused on identifying outliers within the dataset.

## 🛠️ Methodology

The prediction pipeline involves several steps to ensure accuracy:

1.  **Data Cleaning:** Removal of extreme outliers (e.g., impossible floor plans, luxury penthouses) to focus on the standard rental market.
2.  **Feature Engineering:**
    *   **Text Mining:** Extracting keywords like "Fully Furnished," "Gulshan," and "Well Constructed" from the title.
    *   **Math Features:** Creating ratios like `Baths per Bed` and interaction terms like `Area x Premium Area`.
3.  **Encoding:** Target Encoding is used for `Address` and `City` to handle high-cardinality categorical data effectively.
4.  **Modeling:** LightGBM Regressor with regularization (`min_data_in_leaf`, `reg_lambda`) to prevent overfitting.

## 📝 Future Improvements

*   [ ] Integration with live real estate APIs for real-time data fetching.
*   [ ] Support for Sublets and Commercial spaces.
*   [ ] Adding an image analysis feature (e.g., detecting furniture quality from photos).

## 📄 License

This project is open source and available under the MIT License.

---
**Built with ❤️ using Python, LightGBM, and Gradio.**
