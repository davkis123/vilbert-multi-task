#!flask/bin/python
import os

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_pic():
    if request.method == 'POST':
        for e in request.files.keys():
            print(e)
        for e in request.form.keys():
            print(e)

    image = request.files["fileupload"]
    print(image)
    image.save(os.path.join(".", "input_image.jpg"))
    
    caption = request.form["text_input"]
    print(caption)
    text_file = open("input_caption.txt", "w")
    n = text_file.write(caption)
    text_file.close()

    return "ok"

app.run(debug=True)