from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), default='user') # user hoặc admin [cite: 35, 39]
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    result_text = db.Column(db.Text, nullable=False) # Kết quả nhận diện [cite: 31, 33]
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SignData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sign_name = db.Column(db.String(50), nullable=False) # Tên ký hiệu [cite: 36]
    description = db.Column(db.Text)