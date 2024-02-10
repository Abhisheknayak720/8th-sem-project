# from flask import Flask

# app = Flask(__name__)

# @app.route("/members") 
# def members():
#   return "hello"

# if __name__ == "__main__":
#   app.run(debug=True)



# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os

# app = Flask(__name__)
# CORS(app)

# UPLOAD_FOLDER = "uploads"
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file part"})

#     file = request.files["file"]

#     if file.filename == "":
#         return jsonify({"error": "No selected file"})

#     if file:
#         filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#         file.save(filename)

#         # Now you can process the file, save it to MySQL, etc.
#         # Implement your database logic here.

#         return jsonify({"message": "File uploaded successfully"})

#     return jsonify({"error": "Unknown error"})


# if __name__ == "__main__":
#     app.run(debug=True)





from flask import Flask, request, Response
from werkzeug.utils import secure_filename

from db import db_init, db
from models import Img

app = Flask(__name__)
# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///img.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/', methods=['POST'])
def upload():
    pic = request.files['pic']
    if not pic:
        return 'No pic uploaded!', 400

    filename = secure_filename(pic.filename)
    mimetype = pic.mimetype
    if not filename or not mimetype:
        return 'Bad upload!', 400

    img = Img(img=pic.read(), name=filename, mimetype=mimetype)
    db.session.add(img)
    db.session.commit()

    return 'Img Uploaded!', 200


@app.route('/<int:id>')
def get_img(id):
    img = Img.query.filter_by(id=id).first()
    if not img:
        return 'Img Not Found!', 404

    return Response(img.img, mimetype=img.mimetype)