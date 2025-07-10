from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'laika2002'

from app import routes
