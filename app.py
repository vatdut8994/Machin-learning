from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('50_Startups.csv')

le = LabelEncoder()
le.fit(data['State'])
data['State'] = le.transform(data['State'])

x = data.iloc[:, 0: -1]
y = data.iloc[:, -1]

model = LinearRegression()
model.fit(x, y)

app = Flask(__name__)


@app.route('/')
def start():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        rd = int(request.form['R&D value'])
        administration = int(request.form['Administration'])
        market = int(request.form['Market'])
        state = int(request.form['State'])
        usr = {'R&D Spend': [rd], 'Administration': [administration], 'Marketing spend': [market], 'State': [state]}
        usr = pd.DataFrame(usr)
        pred = model.predict(usr)
        return render_template('index.html', profit=pred)


if __name__ == '__main__':
    app.run(debug=True)
