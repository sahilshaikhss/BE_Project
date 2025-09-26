from model.effnet_cbir import EffNetCBIR
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
import random
import string
from PIL import Image

# 🔹 Initialize Flask app
app = Flask(__name__)

# 🔧 Create folders for storing images
os.makedirs("static/individual_photos", exist_ok=True)
os.makedirs("static/group_photos", exist_ok=True)

# 📂 Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///visitors.db'
db = SQLAlchemy(app)

# 🗂️ Visitor model
class Visitor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    guardian_name = db.Column(db.String(100))
    guardian_phone = db.Column(db.String(20))
    address = db.Column(db.String(200))
    members = db.Column(db.String(300))
    unique_code = db.Column(db.String(20), unique=True)
    group_photo = db.Column(db.String(200))

# 📌 Create DB tables (only first time)
with app.app_context():
    db.create_all()

# 🔑 Generate unique 10-char alphanumeric code
def generate_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

# 📝 Registration route
@app.route('/register', methods=['POST'])
def register():
    guardian_name = request.form.get('guardian_name')
    guardian_phone = request.form.get('guardian_phone')
    address = request.form.get('address')
    members = request.form.get('members')  # comma-separated names

    # 📷 Save individual photos
    individual_images = request.files.getlist('individual_photos')
    for i, img in enumerate(individual_images):
        image = Image.open(img)
        image = image.resize((300, 300))  # crop/resize
        image.save(f"static/individual_photos/{guardian_phone}_{i}.jpg")

    # 📷 Save group photo
    group_photo = request.files.get('group_photo')
    group_path = None
    if group_photo:
        group_path = f"static/group_photos/{guardian_phone}_group.jpg"
        group_photo.save(group_path)

    # 📁 Save details to database
    unique_code = generate_code()
    visitor = Visitor(
        guardian_name=guardian_name,
        guardian_phone=guardian_phone,
        address=address,
        members=members,
        unique_code=unique_code,
        group_photo=group_path
    )
    db.session.add(visitor)
    db.session.commit()

    return jsonify({
        "message": "Registration successful",
        "unique_code": unique_code
    }), 201

# 🔹 Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
