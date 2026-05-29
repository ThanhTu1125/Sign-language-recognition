from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    result_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SignData(db.Model):
    __tablename__ = 'signdata' 
    id = db.Column(db.Integer, primary_key=True)
    sign_name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    
    sample_image_path = db.Column(db.String(255)) 
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))

class SignData1(db.Model):
    __tablename__ = 'signdata' 
    id = db.Column(db.Integer, primary_key=True)
    sign_name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)
    
    sample_image_path = db.Column(db.String(255)) 
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    