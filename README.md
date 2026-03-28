# 🌾 Crop Disease Detection System (CNN)

Hi there! 👋 Welcome to the repository for my Bring Your Own Project (BYOP) for my 1st-year B.Tech Computer Science Engineering coursework. 

This project is a complete, end-to-end Machine Learning web application designed to help farmers and agricultural workers identify crop diseases instantly. By simply uploading a picture of a plant leaf, the custom-trained Convolutional Neural Network (CNN) will diagnose the disease, provide a confidence score, and suggest actionable treatments.

## 🚀 Features
* **Custom CNN Architecture:** Built from scratch using TensorFlow/Keras with Dropout layers and Data Augmentation to prevent overfitting.
* **Interactive UI:** A clean, user-friendly web dashboard built with Streamlit.
* **Instant Diagnosis:** Classifies leaves into categories (e.g., Potato Early Blight, Tomato Late Blight, or Healthy) and provides treatment steps.
* **Prediction History:** Automatically saves a log of all previous predictions into a CSV file for tracking.

## 📁 Folder Structure
Here is how the project is organized:
```text
Crop_Disease_Detection/
│
├── app/
│   └── app.py
│
├── data/
│   ├── Potato_Early_blight/
│   ├── Potato_healthy/
│   ├── Tomato_Late_blight/
│   └── Tomato_healthy/
│
├── model/
│   ├── trained_model.h5     # The saved AI "brain" (created after training)
│   ├── accuracy.png         # Graph showing how the AI learned
│   └── loss.png             # Graph showing the AI's error rate
├── notebooks/
│   └── training.py          # The CNN model architecture and training script
├── prediction_history.csv   # Auto-generated log of user predictions
├── requirements.txt         # List of Python libraries needed to run this
└── README.md                # You are reading this right now!
⚙️ Prerequisites
Before running this project, please make sure you have the following installed on your system:
• Python 3.10.x (Highly recommended. Newer versions like 3.13 may cause dependency conflicts with TensorFlow).
• Git (to clone this repository).
• (Windows Users Only): Microsoft Visual C++ Redistributable (Required for TensorFlow's backend C++ operations).
🛠️ Installation Steps
I highly recommend running this inside a Virtual Environment to keep the dependencies clean.
1. Clone the repository:
git clone <https://github.com/dhananjaynaik18>
cd "Crop disease detection"
2. Create a virtual environment:
python -m venv .venv
3. Activate the virtual environment:
Windows:
.venv\Scripts\activate
Mac/Linux:
source .venv/bin/activate
4. Install the required libraries:
pip install -r requirements.txt
🏃‍♂️ How to Run the Project
This project is split into two parts: the training script and the web app.
Part 1: Training the Model (Optional)
Note: I have already included the trained_model.h5 file in the repo, so you can skip this step if you just want to test the web app!
If you want to retrain the model on new data, place your categorized image folders inside the data/ directory, then run:
python notebooks/training.py
This will process the images, train the CNN over 15 epochs (with Early Stopping), and save the new model and graphs into the model/ folder.
Part 2: Running the Web App
To launch the user interface and test the AI, run the Streamlit command:
streamlit run app/app.py
This will start a local server. Open your web browser and go to the Local URL provided in the terminal (usually http://localhost:8501).
🐛 Known Issues & Troubleshooting Notes
During development, I encountered and fixed a few common ML bugs. If you are modifying this code, keep these in mind:
• The "Black Square" Bug: In app.py, do NOT divide the image array by 255.0. The training.py model already has a layers.Rescaling(1./255) layer built into it. Dividing it twice makes the image essentially pitch black, causing the model to blindly guess the first alphabetical class every time.
• Alphabetical Sorting: image_dataset_from_directory sorts folders alphabetically. If you add new diseases, make sure you update the classes list in app.py to match the exact alphabetical order of your data/ folders, or the predictions will be scrambled!