from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

# Creating Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a3f5e6d7c8b9a0d1e2f3a4b5c6d7e8f9'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

# Database instance
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)    
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Import routes at the end to avoid circular imports
from app import routes