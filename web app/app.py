from flask import Flask, render_template, request, jsonify
import json
import pickle
import sys
from lightfm import LightFM
import scipy.sparse as sp
import numpy as np

app = Flask(__name__)
 
restaurant_names = pickle.load(open('restaurant_names.pickle', 'rb'))
id_name_map = pickle.load(open('id_name_map.pickle', 'rb'))
business_map = pickle.load(open('business_map.pickle', 'rb'))
data = pickle.load(open('user_item_coo.pickle', 'rb'))
finalCFmodel = pickle.load(open('finalCFmodel.pickle', 'rb'))
item_labels = pickle.load(open('item_labels.pickle', 'rb')) 
top_5 = pickle.load(open('top_5.pickle', 'rb'))  
CF_model = pickle.load(open('CF_model.pickle', 'rb'))  

def get_recommendation(user_input, top_5 = top_5, model=CF_model, data=data, item_labels=item_labels):
	'''
	Args:
		user_input (str): restaurant(s) that user has already visited
		top_5: top 5 restaurants to return in case user_input_list = None
		model: Collaborative filtering model built using LightFM
		data: user_item sparse matrix used for model construction
		user_index: default to zero for new user that was not in training data
	Returns: 
		strings: restaurants to recommend user 
	'''
	if user_input is not "":

		def get_b_indices(user_input, id_name_map = id_name_map, business_map = business_map):
		    
		    def get_key(id_name_map, values):
		        return list(id_name_map.keys())[list(id_name_map.values()).index(values)]
		    
		    def get_index(index_map, key):
		        return index_map[key]

		    b_ids = get_key(id_name_map, user_input)
		    #b_ids = []
		    #for i in user_input:
		    #    b_ids.append(get_key(id_name_map, i))
		        
		    b_indices = get_index(business_map, b_ids)
		    #for i in b_ids:
		    #    b_indices.append(get_index(business_map, i))
		    
		    return (b_indices)

		# get indices of restaurants that user has been to in order to construct feature matrix
		user_input_indices = get_b_indices(user_input)


		def build_new_user_features(user_input_indices, data = data):

		    dok = sp.dok_matrix((1, data.shape[1]), dtype = np.float32)
		    
		    #for i in user_input_indices:
		    dok[0, user_input_indices] = 1
		    
		    return dok.tocsr()


		# construct feature matrix to feed into LightFM's predict function 
		new_user_features = build_new_user_features(user_input_indices)


		# get score of all items in training set
		scores = model.predict(0, item_ids = np.arange(data.shape[1]), user_features = new_user_features)

		# get indices of top item
		top_items = item_labels[np.argsort(-scores)]

		return top_items[:5]

	else: 
		return top_5


@app.route('/')
def index():
   return render_template('index.html')
 
@app.route('/result', methods = ['POST'])
def result():
	
	user_input = request.form.get('restaurant')
	prediction = get_recommendation(user_input)
	return render_template("result.html",result = prediction)

 
if __name__ == '__main__':
   app.run(debug = True)