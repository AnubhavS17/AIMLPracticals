from flask import Flask, render_template, request
import iris_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def basic():
   if request.method == 'POST':
       sepal_length = float(request.form['sepallength'])
       sepal_width = float(request.form['sepalwidth'])
       petal_length = float(request.form['petallength'])
       petal_width = float(request.form['petalwidth'])

       y_pred = [[sepal_length, sepal_width, petal_length, petal_width]]
       trained_model = iris_model.training_model()
       prediction_value = trained_model.predict(y_pred)[0]

       if prediction_value == 0:
           return render_template('index.html', result="The flower is classified as Setosa 🌸")
       elif prediction_value == 1:
           return render_template('index.html', result="The flower is classified as Versicolor 🌿")
       else:
           return render_template('index.html', result="The flower is classified as Virginica 🌺")

   return render_template('index.html')

if __name__ == '__main__':
   app.run(debug=True)
