import os
from flask import render_template, url_for, flash, redirect, request, abort,Flask
from frontend import app
from werkzeug.utils import secure_filename
from classifier import testing


posts=[12,22,11,14,15,1,51,21,12,432,534,3453,2,24,23,4,23,4,2,34,2,3,31,314]

# not to remove other images
#os.remove('frontend/static/importance.png')


@app.route("/")
@app.route("/home", methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      s = f.filename
      x = testing(s)
      return render_template("home.html",posts=x,src="/static/cm.png")
    return render_template("home.html",posts=[],src="/static/temp.png")

@app.route("/importantChart")
def importantChart():
  return render_template("importantChart.html")

