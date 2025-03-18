from flask import Flask, request, render_template, url_for, redirect, flash
import secrets
import io
from PIL import Image
import pickle
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask_login import login_user, current_user, logout_user, login_required
from app import app, db, bcrypt
from app.forms import RegistrationForm, LoginForm, UpdateAccountForm
from app.models import User, Post

# Load ML models and scalers
try:
    model = pickle.load(open('model.pkl', 'rb'))  # Crop prediction model
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
    dtr = pickle.load(open('dtr.pkl', 'rb'))  # Decision Tree Regressor for yield prediction
    preprocesser = pickle.load(open('preprocesser.pkl', 'rb'))
except FileNotFoundError as e:
    raise RuntimeError(f"Required file is missing: {e}")

# Load the trained TensorFlow model for plant disease prediction
plant_disease_model = tf.keras.models.load_model("trained_plant_disease_model.keras")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'

# Helper function to check if a file extension is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template("home.html")

# Crop prediction route
@app.route("/predict_crop", methods=['GET', 'POST'])
def predict_crop():
    try:
        N = float(request.form.get('Nitrogen', 0))
        P = float(request.form.get('Phosphorus', 0))
        K = float(request.form.get('Potassium', 0))
        temp = float(request.form.get('Temperature', 0))
        humidity = float(request.form.get('Humidity', 0))
        ph = float(request.form.get('Ph', 0))
        rainfall = float(request.form.get('Rainfall', 0))

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply transformation
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = model.predict(final_features)

        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
                     6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
                     11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
                     16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
                     20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        crop = crop_dict.get(prediction[0], "Unknown crop")
        result = f"{crop} is the best crop to be cultivated right there."
    except Exception as e:
        result = f"Error: {e}"
    return render_template('index.html', result=result)

# Crop yield prediction route
@app.route("/predict_yield", methods=['GET', 'POST'])
def predict_yield():
    prediction_value = None

    if request.method == 'POST':
        try:
            Year = int(request.form['Year'])
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']
            Item = request.form['Item']

            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
            transformed_features = preprocesser.transform(features)

            prediction_value = dtr.predict(transformed_features).reshape(1, -1)[0][0]

        except Exception as e:
            prediction_value = f"Error: {str(e)}"

    return render_template('yield.html', prediction=prediction_value, title="Yield Prediction")

# Image preprocessing function for plant disease prediction
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch format
    input_arr = input_arr / 255.0  # Normalize pixel values
    return input_arr

# Plant disease prediction route
@app.route('/predict_disease', methods=['GET', 'POST'])
def predict_disease():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image and predict
            processed_image = preprocess_image(filepath)
            predictions = plant_disease_model.predict(processed_image)
            result_index = np.argmax(predictions)
            
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            prediction = class_names[result_index]
            image_path = url_for('static', filename=f'uploads/{filename}')

            return render_template('result.html', prediction=prediction, image_path=image_path)
        else:
            flash('Invalid file format. Please upload a PNG, JPG, or JPEG file.', 'danger')
            return redirect(request.url)

    return render_template('disease.html')

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)


@app.route("/register",methods=['GET','POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('index'))
    return render_template('register.html',title = 'Register',form = form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():  # This checks if the form is submitted and validated
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!', 'success')  # Corrected 'siccess' to 'success'
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please Check username and Password', 'danger')  # Corrected 'Unsccessful' to 'Unsuccessful'
    return render_template('login.html', title='Login', form=form)

# Logout route
@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route("/account",methods=['GET','POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!','success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static',filename='profile_pics/'+ current_user.image_file)
    return render_template('account.html', title='Account', image_file = image_file, form=form)