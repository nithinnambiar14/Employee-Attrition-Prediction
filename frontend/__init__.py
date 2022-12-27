# Flask library are imported here
import os
from flask import Flask
from flask_caching import Cache


#Define app name
app = Flask(__name__)
#cache=Cache(app,config={'CACHE_TYPE':'simple'})
app.config['CACHE_TYPE']='null'
app.cache=Cache(app)



#Set up key
app.config['SECRET_KEY']='1a858e5d5f93ac4338efe291b34f9d8c'
app.config['UPLOAD_FOLDER']= ""



#call home page
from frontend import routes

