from flask import Flask, request, render_template
from sklearn.externals import joblib
import json
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/api',methods=['POST','GET'])
def predict_attrition():
    if request.method=='POST':
        result = request.form
        
        #Prepare the feature vector for prediction
        
        #index_dict = joblib.load("indext_dict.pkl")

        
        age= result['age']
        distance_from_home = result['distance_from_home']
        environment_satisfaction = result['environment_satisfaction']
        job_involvement = result['job_involvement']
        job_satisfaction = result['job_satisfaction']
        monthly_income = result['monthly_income']
        num_companies_worked = result['num_companies_worked']
        performance_rating = result['performance_rating']
        relationship_satisfaction = result['relationship_satisfaction']
        total_working_years = result['total_working_years']
        training_times_last_year = result['training_times_last_year']
        work_life_balance = result['work_life_balance']
        years_at_company = result['years_at_company']
        years_in_current_role = result['years_in_current_role']
        years_since_last_promotion = result['years_since_last_promotion']
        years_with_curr_manager = result['years_with_curr_manager']
        
        



        user_input = {'age':age,
                      'distance_from_home' : distance_from_home,
                      'environment_satisfaction' : environment_satisfaction,
                      'job_involvement' : job_involvement,
                      'job_satisfaction' : job_satisfaction,
                      'monthly_income' : monthly_income,
                      'num_companies_worked' : num_companies_worked,
                      'performance_rating' : performance_rating,
                      'relationship_satisfaction' : relationship_satisfaction,
                      'total_working_years' : total_working_years,
                      'training_times_last_year' : training_times_last_year,
                      'work_life_balance' : work_life_balance,
                      'years_at_company' : years_at_company,
                      'years_in_current_role' : years_in_current_role,
                      'years_since_last_promotion' : years_since_last_promotion,
                      'years_with_curr_manager' : years_with_curr_manager

        }

        a = np.array(list(user_input.values()))
        a = a.reshape(1,-1)
 
        rf = joblib.load("random_forest_model.pkl")
        prediction = int(rf.predict(a))
        return json.dumps({'prediction':prediction});
        
        #return render_template('result.html',prediction = prediction)

    
if __name__ == '__main__':
    app.debug = True
    app.run()