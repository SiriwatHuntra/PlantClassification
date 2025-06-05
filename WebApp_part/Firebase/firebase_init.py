import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase

import os

# Set GOOGLE_APPLICATION_CREDENTIALS environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(".json")

def initialize_firebase_admin():
    if not firebase_admin._apps:
        cred = credentials.Certificate(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        app = firebase_admin.initialize_app(
            cred,
            {
                "storageBucket": ""
            }
        )
        db = firestore.client()
        return cred, app, db
    else:
        app = firebase_admin.get_app()
        db = firestore.client()
        return None, app, db

# Initialize Pyrebase (for Authentication and Realtime Database)
def initialize_pyrebase():
    return pyrebase.initialize_app(firebaseConfig)

# Firebase configuration
firebaseConfig = {
    "apiKey": "",
    "authDomain": "",
    "projectId": "",
    "storageBucket": "",
    "messagingSenderId": "",
    "appId": "",
    "measurementId": "",
    "databaseURL": ""
}

# Google OAuth credentials
GOOGLE_CLIENT_ID = ""
GOOGLE_CLIENT_SECRET = ""
GOOGLE_REDIRECT_URI = ""
