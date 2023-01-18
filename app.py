from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'logres.pkl'
#model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    parental_level_of_education = request.form['parental_level_of_education']
    lunch = request.form['lunch']
    test_preparation_course = request.form['test_preparation_course']
    math_score = request.form['math_score']
    reading_score = request.form['reading_score']
    writing_score = request.form['writing_score']

    
      
    pred = model.predict(np.array([[gender,parental_level_of_education,lunch,test_preparation_course,math_score,reading_score,writing_score]]))
    print(pred)
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    app.run
