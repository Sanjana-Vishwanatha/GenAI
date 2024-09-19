import random
from flask import Flask, request, jsonify, Response, make_response
from flask_cors import CORS
import pandas as pd
import json
import numpy as np
import os
from flask import Flask, request, jsonify
import pandas as pandas
from pickle import dump,load
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from tensorflow import keras
# from keras.models import Model
# from keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime
from pymongo import MongoClient
import numpy as np
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import ReLU
from tensorflow.keras.metrics import MeanSquaredError

import time
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import pickle
import openai

# Load the environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

useHardCodedData = os.getenv('USE_HARDCODED_DATA', default=True)
current_model = None

def genai_sucess_rate(prompt):
    print(f"generating text based output using GENAI..GPT 3.5")
    openai.api_key = os.getenv('OPENAI_API_KEY') 
    # client = OpenAI(os.getenv('OPENAI_API_KEY'))

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Write a good morning greeting message"
            }
        ]
    )

    print(completion.choices[0].message)


def scale_with_pickle_file(action, data, questionSlugAndType, model_name, model_dir):
    
    # Convert the dictionary to a Pandas DataFrame
    if str(action).lower() == 'predict':
        df = pd.DataFrame([data])

    elif str(action).lower() == 'train':
        df = pd.DataFrame(data)

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Create and Ensure the output directory exists to save the train output.
    dir = os.path.join(model_dir, 'pickle_output')
    if not os.path.exists(dir):
        os.makedirs(dir)

    if str(action).lower() == 'train':
        
        # Fit and transform the data using questionSlugAndType
        with open(os.path.join(model_dir, 'questionSlugAndType.json'), 'r') as json_file:
            total_questions = json.load(json_file)

        # Add any missing columns in prediction data with default value 0
        for key in total_questions.keys():
            if key not in df.columns:
                df[key] = 0
        print(f"After adding missing columns: {df}")

        file_path = os.path.join(dir, "train_pickle.json")
        # Writing JSON data to the file
        with open(file_path, 'w') as json_file:
            json.dump(df.to_dict(), json_file, indent=4)

        print(f"JSON data has been saved to {file_path}")
        scaled_data = scaler.fit_transform(df)

        # Save the fitted scaler
        scaler_file = os.path.join(model_dir, model_name + '_scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)

    elif str(action).lower() == 'predict':
        # Load the fitted scaler
        scaler_file = os.path.join(model_dir, model_name + '_scaler.pkl')
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)

        # Load questionSlugAndType from JSON file to get the correct feature order
        with open('models/model_test_new/pickle_output/train_pickle.json', 'r') as f:
          train_data = json.load(f)
        
        print(type(train_data))  # Check if it's a list, dict, etc.

        # Assuming train_data is a list of dictionaries (common structure), adjust as follows:
        if isinstance(train_data, list) and len(train_data) > 0:
            # Extract column names assuming train_data is a list of dictionaries
            train_columns = list(train_data[0].keys())
        elif isinstance(train_data, dict):
            # If train_data is a dictionary, directly get the keys
            train_columns = list(train_data.keys())
        else:
            raise ValueError("Unexpected data structure in train_pickle.json")


        # Add any missing columns in prediction data with default value 0
        for key in train_columns:
            if key not in df.columns:
                df[key] = 0

        print("Columns before filtering:", df.columns)
        df = df[train_columns]
        print("Columns after filtering:", df.columns)

        # Scale the data using the loaded scaler
        scaled_data = scaler.transform(df)

        print(scaled_data)

    # Convert the scaled data back to a DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    print(f"Scaled data dataframe {scaled_df}")

    def success_rate():
      # Assigning success rate using the priority given.
      priority_scores = {key: value['priorityScore'] for key, value in questionSlugAndType.items()}
      total_priority_score = sum(priority_scores.values())

      # Normalize the priority scores to get the weights so that their sum equals 100
      weights = {key: (score / total_priority_score) * 100 for key, score in priority_scores.items()}

      # Calculate the success rate as a percentage
      scaled_df['success_rate'] = 0  # Initialize the column with zeros
      for col, weight in weights.items():
          scaled_df['success_rate'] += scaled_df[col] * weight

      # Ensure the success rate is between 0 and 100
      scaled_df['success_rate'] = scaled_df['success_rate'].clip(0, 100)
      
      print(f"After adding the Success Rate columns {scaled_df}")


    if str(action).lower() == 'train':
        data_order = list(questionSlugAndType.keys())

        all_data_columns = set(df.columns)
        missing_cols = set(data_order) - all_data_columns

        # Assigning null values to the empty columns.
        for c in missing_cols:
            scaled_df[c] = 0

        # print(f"After adding the missed columns {scaled_df}")

        success_rate()
        x = scaled_df[list(data_order)]
        y = scaled_df["success_rate"]
        X_scaled = scaler.fit_transform(x)
        return X_scaled, y
    
    elif str(action).lower() == 'predict':
        return scaled_df

# Function to convert answer to its corresponding weightage
def convert_to_weightage(answer, question):
    options = question.get('options', [])
    for option in options:
        if answer == option['optionValue']:
            return option['optionWeightage']
    return None

def predict_neural_model():
    action='predict'

    #starting the timer
    start_time = time.time()

    if useHardCodedData:
       file_path = 'models/model_test_new/predict_seeded_data.json'
       # Reading the JSON file
       with open(file_path, 'r') as file:
        data = json.load(file)
    else:
        data = request.json

    # Extract the model path
    model_path = data['modelDetails']['filePath']
    model_name = data['modelDetails']['name']
    model_dir = os.path.join('models', model_name)

    # Extract the allAnswers dictionary
    all_answers = data['allAnswers']

    # Iterate over all the answers
    for answer in data['allAnswers']:
        for key, value in answer.items():
            if isinstance(value, dict):
                # Replace the value with the weight if it's a dictionary
                answer[key] = value['weight']
            elif isinstance(value, list):
                # Sum the weights if it's a list and replace the entry with the sum
                answer[key] = sum(item['weight'] for item in value)

    # Save the updated data to a JSON variable
    updated_json_data = json.dumps(data, indent=4)

    print(f"updated_json_data: {updated_json_data}")

    data_dict = json.loads(updated_json_data)

    # Extract the `allAnswers` data
    answers = data_dict['allAnswers']

    # Flatten the `allAnswers` list of dictionaries into a single dictionary
    flattened_answers = {}
    for answer in answers:
        flattened_answers.update(answer)

    print(flattened_answers)
    # Convert the flattened dictionary into a DataFrame
    df = pd.DataFrame([flattened_answers])
    print(df)
    # Extract only the questions (keys)
    data_dict = json.loads(updated_json_data)
    questions = []
    for answer in data_dict['allAnswers']:
        questions.extend(answer.keys())

    #scaling the answers between 0 and 1
    scaled_data = scale_with_pickle_file(action, flattened_answers, questions, model_name, model_dir)

    # Load the model
    model = load_model(model_path)

    # Print the DataFrame
    print(f"scaled data: {scaled_data}")

    # Predict the result
    predictions = model.predict(scaled_data)

    # End timing
    end_time = time.time()

    # Calculate the elapsed time
    total_time = end_time - start_time

    # Round and calculate the success rate
    # Scale the prediction to a percentage
    # predicted_successRate = min(max(predictions[0][0], 0), 1) * 100
    # predicted_successRate = min(max(predictions[0][0] * 10, 0), 100)  # Ensure the value is between 0 and 100

    predicted_successRate = round(predictions[0][0] / 10, 2)
    # predicted_successRate = round((predictions[0][0],2)*100).tolist()

    output = {
    'Success_Rate': predicted_successRate,
    'timeTook': [total_time]
    }

    # Check environment variable
    model_env = os.getenv('MODELENV', 'development')

    if model_env == 'production' or model_env == 'development':
        # Return as a request for production
        return make_response(jsonify(output), 200)
    else:
        # we might want to send the make_response(jsonify(output), 200) and json.dumps(output)
        # Return JSON output for development
        return json.dumps(output)

def trainNeuralModel_with_request_options():
    start_time = time.time()
    action = 'train'
    # Load the JSON file
    if useHardCodedData:
      file_path = 'models/model_test_new/training_seeded_data.json'
      # Reading the JSON file
      with open(file_path, 'r') as file:
        data = json.load(file)
    else:
        data = request.json
    # Extracting the question details with their respective weightages
    questions = data['data']['questionSlugAndType'][0]['questions']

    model_name = data['data']['modelName']
    model_dir = os.path.join('models', model_name)

    # Save questionSlugAndType to a JSON file for future use.
    with open(os.path.join(model_dir, 'questionSlugAndType.json'), 'w') as json_file:
        json.dump(questions, json_file)

    #getting the model name
    model_name = data['data']['modelName']

    # Extracting the answers and converting them to weightages
    all_answers_with_weightages = []
    for response in data['data']['answers']:
        weighted_answers = {}
        for key, answer in response['answers'].items():
            question = questions.get(key)
            if question:
                weightage = convert_to_weightage(answer, question)
                weighted_answers[key] = weightage if weightage is not None else answer
        all_answers_with_weightages.append(weighted_answers)
    
    print(f"All answers has been updated to its corresponding weitages: {all_answers_with_weightages}")

    #Scaling the data
    x,y=scale_with_pickle_file(action, all_answers_with_weightages, questions, model_name, model_dir)

    print(x)
    print(y)

    #Neural model training starts

    traindata_x,testdata_x,traindata_y,testdata_y=train_test_split(x,y,test_size=0.2, random_state=42)
    print(traindata_y,testdata_y)
    model=Sequential([Dense(50, input_shape=(x.shape[1],),activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(1,activation='relu')])
    model.compile(optimizer='adam',loss='mean_squared_error' ,metrics=['mean_squared_error'])
    model.summary()
    model_fit=model.fit(x=traindata_x,y=traindata_y,epochs=50,batch_size=10, validation_split=0.2)
    model_validate=model.predict(testdata_x)
    mae=mean_absolute_error(testdata_y,model_validate)

    #edited........
    mse_metric = MeanSquaredError()
    mse_metric.update_state(testdata_y, model_validate)
    mse = mse_metric.result().numpy()
    r2_sc=r2_score(testdata_y,model_validate)

    # Saving the model
    model_path = os.path.join(model_dir, f'{model_name}.h5')
    model.save(model_path)

    # Get the absolute file path
    file_path = os.path.abspath(f'{model_name}.h5')

    # End timing
    end_time = time.time()

    # Calculate the elapsed time
    total_time = end_time - start_time
    output = {
    'absolute': mae.tolist(),  # Convert NumPy array to a Python list
    'squared': [float(mse)],
    'r2': [float(r2_sc)] ,
    'trainTime': [total_time],
    'filePath' :[model_path]
    }

    # Check environment variable
    model_env = os.getenv('MODELENV', 'local')

    if model_env == 'production' or model_env == 'development':
        # Return as a request for production
        return make_response(jsonify(output), 200)
    else:
        # Return JSON output for development
        return json.dumps(output)

def train_seedData_with_given_options(input, model_dir):
    start_time = time.time()
    action = 'train'
    # Load the JSON file
    if useHardCodedData:
      file_path = 'models/model_test_new/training_seeded_data.json'
      # Reading the JSON file
      with open(file_path, 'r') as file:
        data = json.load(file)
    else:
        data = input
    # Extracting the question details with their respective weightages
    questions = data['data']['questionSlugAndType'][0]['questions']

    # Save questionSlugAndType to a JSON file for future use.
    with open(os.path.join(model_dir, 'questionSlugAndType.json'), 'w') as json_file:
        json.dump(questions, json_file)

    #getting the model name
    model_name = data['data']['modelName']

    # Extracting the answers and converting them to weightages
    all_answers_with_weightages = []
    for response in data['data']['answers']:
        weighted_answers = {}
        for key, answer in response['answers'].items():
            question = questions.get(key)
            if question:
                weightage = convert_to_weightage(answer, question)
                weighted_answers[key] = weightage if weightage is not None else answer
        all_answers_with_weightages.append(weighted_answers)
    
    print(f"All answers has been updated to its corresponding weitages: {all_answers_with_weightages}")

    #Scaling the data
    x,y=scale_with_pickle_file(action, all_answers_with_weightages, questions, model_name, model_dir)

    print(x)
    print(y)

    #Neural model training starts

    traindata_x,testdata_x,traindata_y,testdata_y=train_test_split(x,y,test_size=0.2, random_state=42)
    print(traindata_y,testdata_y)
    model=Sequential([Dense(50, input_shape=(x.shape[1],),activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(50,activation='relu'),
                 Dropout(0.1),
                 Dense(1,activation='relu')])
    model.compile(optimizer='adam',loss='mean_squared_error' ,metrics=['mean_squared_error'])
    model.summary()
    model_fit=model.fit(x=traindata_x,y=traindata_y,epochs=50,batch_size=10, validation_split=0.2)
    model_validate=model.predict(testdata_x)
    mae=mean_absolute_error(testdata_y,model_validate)

    #edited........
    mse_metric = MeanSquaredError()
    mse_metric.update_state(testdata_y, model_validate)
    mse = mse_metric.result().numpy()
    r2_sc=r2_score(testdata_y,model_validate)

    # Saving the model
    model_path = os.path.join(model_dir, f'{model_name}.h5')
    model.save(model_path)

    # Get the absolute file path
    file_path = os.path.abspath(f'{model_name}.h5')

    # End timing
    end_time = time.time()

    # Calculate the elapsed time
    total_time = end_time - start_time
    output = {
    'absolute': mae.tolist(),  # Convert NumPy array to a Python list
    'squared': [float(mse)],
    'r2': [float(r2_sc)] ,
    'trainTime': [total_time],
    'filePath' :[model_path]
    }

    # Check environment variable
    model_env = os.getenv('MODELENV', 'local')

    if model_env == 'production' or model_env == 'development':
        # Return as a request for production
        return make_response(jsonify(output), 200)
    else:
        # Return JSON output for development
        return json.dumps(output)

def generate_seed_data(data):
    seed_qty = int(data["seedQty"])
    questionSlugAndType = data["questionSlugAndType"][0]
    questions = questionSlugAndType["questions"]
    answers_list = []
    for _ in range(seed_qty):
        answers = {}
        for question, details in questions.items():
            if "seedData" not in details or not details["seedData"]:
                continue
            seed_data = details["seedData"]
            seed_values = [item["seedValue"] for item in seed_data]
            seed_probabilities = [float(item["seedProbability"]) for item in seed_data]
            selected_value = random.choices(seed_values, seed_probabilities)[0]
            answers[question] = selected_value
        answers_list.append({
            "answers": answers
        })
    return {
        "data": {
            "answers": answers_list
        }
    }

def train_with_seed_data(data):
    # Generate seed data
    seed_data = generate_seed_data(data['data'])

    # Update the original data with generated seed data
    data['data']['answers'] = seed_data['data']['answers']

    model_name = data['data']['modelName']
    model_dir = os.path.join('models', model_name)
    os.makedirs(model_dir, exist_ok=True)
    training_data_path = os.path.join(model_dir, 'SeededTrainingData.json')
    with open(training_data_path, 'w') as f:
        json.dump(data, f, indent=2)

    return train_seedData_with_given_options(data, model_dir)

@app.route('/seed', methods=['POST'])
def trainSeededModel_with_options_route():
    print("Training the model when options are given..")
    if useHardCodedData:
       file_path = 'models/model_test_new/training_data.json'
       # Reading the JSON file
       with open(file_path, 'r') as file:
        data = json.load(file)

    else:
        data = request.json

    return train_with_seed_data(data)

@app.route('/trainNeural', methods=['POST'])
def train_neural_route_from_request():
    print("Training the model")
    print(request.json)
    return trainNeuralModel_with_request_options()

@app.route('/predict', methods=['POST'])
def predict_neural_route():
    print("Predicting the model")

    return predict_neural_model()
if __name__ == '__main__':
    user_prompt = "Write a short story about a Large Level Models"
    output = genai_sucess_rate(user_prompt)
    print(output)
    # print(trainSeededModel_with_options_route())
    # print(predict_neural_route())
    model_env = os.getenv('MODELENV', 'local')
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5500))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    print(f"Model environment: {model_env}")
    app.run(host=host, port=port, debug=debug)