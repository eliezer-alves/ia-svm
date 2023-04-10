from flask import Flask, jsonify, request, render_template
from datetime import date
import pickle

app = Flask(__name__, template_folder="views")

with open('modelo.sav', 'rb') as file:
    knn = pickle.load(file)

with open('modelosvm.sav', 'rb') as file:
    svclassifier = pickle.load(file)


# WEB ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/iris-knn', methods=['POST', 'GET'])
def web_iris_knn():

    data = {
        "length_s": 0,
        "width_s": 2,
        "length_p": 0,
        "width_p": 0,
        "result": 0
    }

    if request.method == 'POST':
        length_s = float(request.form.get('length_s'))
        width_s = float(request.form.get('width_s'))
        length_p = float(request.form.get('length_p'))
        width_p = float(request.form.get('width_p'))
        
        y_pred = knn.predict([[length_s, width_s, length_p, width_p]])
    
        r = ""
        if y_pred[0] == 0:
            r = "Iris-setosa"
        elif y_pred[0] == 1:
            r = "Iris-versicolor"
        elif y_pred[0] == 2:
            r = "Iris-virginica"

        data = {
            "length_s": length_s,
            "width_s": width_s,
            "length_p": length_p,
            "width_p": width_p,
            "result": r
        }

    return render_template('iris-knn.html', data=data, url=request.base_url)

@app.route('/banknote-svm', methods=['POST', 'GET'])
def web_banknote_svm():

    data = {
        "variance": 0,
        "skewness": 2,
        "curtosis": 0,
        "entropy": 0,
        "result": 0
    }

    if request.method == 'POST':
        variance = float(request.form.get('variance'))
        skewness = float(request.form.get('skewness'))
        curtosis = float(request.form.get('curtosis'))
        entropy = float(request.form.get('entropy'))
        
        y_pred = svclassifier.predict([[variance, skewness, curtosis, entropy]])
    
        r = ""
        if y_pred[0] == 0:
            r = "Iris-setosa"
        elif y_pred[0] == 1:
            r = "Iris-versicolor"
        elif y_pred[0] == 2:
            r = "Iris-virginica"

        data = {
            "variance": variance,
            "skewness": skewness,
            "curtosis": curtosis,
            "entropy": entropy,
            "result": r
        }

    return render_template('banknote-svm.html', data=data, url=request.base_url)




# API ROUTES
@app.route('/api/hello')
def hello():
    return jsonify({
        "greeting": ["hello", "world"],
        "date": date.today()
    })

@app.route('/api/iris-knn', methods=['POST'])
def api_iris_knn():
    length_s = float(request.form.get('length_s'))
    width_s = float(request.form.get('width_s'))
    length_p = float(request.form.get('length_p'))
    width_p = float(request.form.get('width_p'))
    
    y_pred = knn.predict([[length_s, width_s, length_p, width_p]])
   
    r = ""
    if y_pred[0] == 0:
        r = "Iris-setosa"
    elif y_pred[0] == 1:
        r = "Iris-versicolor"
    elif y_pred[0] == 2:
        r = "Iris-virginica"

    return jsonify({
        "length_s": length_s,
        "width_s": width_s,
        "length_p": length_p,
        "width_p": width_s,
        "result": r
    })

@app.route('/api/banknote/svm', methods=['POST'])
def api_banknotes_svm():
    variance = float(request.form.get('variance'))
    skewness = float(request.form.get('skewness'))
    curtosis = float(request.form.get('curtosis'))
    entropy = float(request.form.get('entropy'))

    pred = svclassifier.predict([[variance, skewness, curtosis, entropy]])[0]
    result = "A nota eh falsa"

    if pred == 0 :
        result = "A nota eh verdadeira"

    return jsonify({
        "result": result
    })
    

if __name__ == '__main__':
    app.run()

