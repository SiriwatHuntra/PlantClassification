from flask import Flask, render_template, request, redirect, url_for, session, flash
from Firebase.firebase_init import initialize_firebase_admin, initialize_pyrebase
import uuid, io
from functools import wraps
from Module.Image import background_removal
from Module.Classifier import image_classify
from firebase_admin import storage
from datetime import datetime
import firebase
from datetime import datetime
from flask import request


app = Flask(__name__, template_folder='template')
app.secret_key = "your-secret-key_1260"

# Initialize Firebase
cred,_, db = initialize_firebase_admin()  # For Firestore operations
firebase = initialize_pyrebase()               # For Pyrebase (Auth)
auth = firebase.auth()                         # Firebase Authentication

# Decorator to protect routes
@app.context_processor
def inject_user_role():
    """Inject user role into templates for rendering admin-specific options."""
    user_email = session.get('user')
    if user_email:
        user_docs = db.collection('users').where('email', '==', user_email).limit(1).get()
        if user_docs:
            role = user_docs[0].to_dict().get('role', 'user')
            return {'role': role}
    return {'role': None}

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        # Check if the user is logged in by verifying 'user' in session
        if 'user' not in session or not session.get('user'):
            # Redirect to the login page if not authorized
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

def role_required(role):
    def decorator(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            user_email = session.get('user')
            if not user_email:
                return redirect(url_for('login'))

            user_docs = db.collection('users').where('email', '==', user_email).limit(1).get()
            if not user_docs or user_docs[0].to_dict().get('role') != role:
                return "Unauthorized Access", 403

            return f(*args, **kwargs)
        return wrap
    return decorator

# Helper function to handle Firebase errors
def handle_firebase_error(e):
    error_message = str(e)
    if "EMAIL_EXISTS" in error_message:
        return "This email is already registered."
    elif "INVALID_EMAIL" in error_message:
        return "Invalid email format."
    elif "WEAK_PASSWORD" in error_message:
        return "Password should be at least 6 characters."
    return "An error occurred. Please try again."

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    user_email = session.get('user')
    if not user_email:
        return redirect(url_for('login'))

    try:
        users_ref = db.collection('users')
        user_docs = users_ref.where('email', '==', user_email).limit(1).get()

        if not user_docs:
            return "User not found. Please log in again.", 400

        user_doc = user_docs[0]
        user_uid = user_doc.id
        user_data = user_doc.to_dict()
        username = user_data.get('user_name', 'Unknown')

        last_log = None 
        latest_image = None

        if request.method == 'POST' and 'upload_image' in request.form:
            # Handle image upload
            image = request.files.get('image')
            if not image:
                return render_template('Home.html', username=username, email=user_email, error="No image uploaded.")
            
            # Generate unique identifier for the image
            image_uid = str(uuid.uuid4())

            # Step 1: Background removal
            bgrm_image, classify_img = background_removal(image)

            # Step 2: Image classification
            classification_results = image_classify(classify_img)

            # Step 3: Upload image to Firebase Storage
            firebase_path = f"users/{user_uid}/images/{image_uid}.jpg"
            bucket = storage.bucket()
            blob = bucket.blob(firebase_path)

            result_img_byte_array = io.BytesIO()
            bgrm_image.save(result_img_byte_array, format='JPEG')
            result_img_byte_array.seek(0)

            blob.upload_from_string(result_img_byte_array.getvalue(), content_type="image/jpeg")
            blob.make_public()

            # Step 4: Log metadata in Firestore
            image_data = {
                'image_uid': image_uid,
                'path': blob.public_url,
                'user_uid': user_uid,
                'upload_time': datetime.utcnow(),
                'classification': classification_results
            }
            logs_ref = db.collection('users').document(user_uid).collection('logs')
            logs_ref.document(image_uid).set(image_data)

            last_log = image_data
            latest_image = {'name': image_uid, 'path': blob.public_url}

            return render_template('Home.html', username=username, email=user_email, message="Image uploaded successfully!",
                                   latest_image=latest_image, last_log=last_log)

        return render_template('Home.html', username=username, email=user_email, latest_image=None, last_log=None)

    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred. Please try again later.", 500


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        ip_address = request.remote_addr  # Get the user's IP address
        user_agent = request.headers.get('User-Agent')  # Get the user agent (browser/device info)

        try:
            # Authenticate the user with Firebase Authentication
            user = auth.sign_in_with_email_and_password(email, password)

            # Save user email in the session
            session['user'] = email

            # Get the user document from Firestore
            user_docs = db.collection('users').where('email', '==', email).limit(1).get()
            if not user_docs:
                raise ValueError("User document not found in Firestore.")

            user_doc = user_docs[0]
            user_id = user_doc.id

            # Log the successful login
            log_data = {
                'timestamp': datetime.utcnow(),
                'ip_address': ip_address,
                'status': 'Success',
                'user_agent': user_agent
            }
            db.collection('users').document(user_id).collection('login_logs').add(log_data)

            return redirect(url_for('home'))

        except Exception as e:
            print(f"Login error: {e}")

            # Log the failed login
            failed_log_data = {
                'timestamp': datetime.utcnow(),
                'ip_address': ip_address,
                'status': 'Failed',
                'user_agent': user_agent
            }
            user_docs = db.collection('users').where('email', '==', email).limit(1).get()
            if user_docs:
                user_doc = user_docs[0]
                user_id = user_doc.id
                db.collection('users').document(user_id).collection('login_logs').add(failed_log_data)

            return render_template('Login.html', error="Login failed. Invalid username or password.")

    return render_template('Login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        #value field
        email = request.form['email'].strip()
        password = request.form['password']
        username = request.form['username'].strip()
        admin_key = request.form.get('admin_key').strip()

        # Validate input
        if not email or not password or not username:
            return render_template('Registering.html', error="All fields are required.")

        if len(password) < 6:
            return render_template('Registering.html', error="Password must be at least 6 characters long.")

        #Admin register key
        ADMIN_KEY = 'Admin'
        #User role
        role = "admin" if admin_key == ADMIN_KEY else "user"
        if role == "admin" and admin_key != ADMIN_KEY:
            return render_template('Registering.html', error="Invalid admin key.")
        
        try:
            # Check for invalid email format
            if '@' not in email or '.' not in email:
                return render_template('Registering.html', error="Invalid email format.")

            # Check for duplicate username or email in Firestore
            users_ref = db.collection('users')
            existing_user_email = users_ref.where('email', '==', email).get()
            if existing_user_email:
                return render_template('Registering.html', error="This email is already registered.")

            existing_user_name = users_ref.where('user_name', '==', username).get()
            if existing_user_name:
                return render_template('Registering.html', error="This username is already taken.")

            # Create a new user in Firebase Authentication
            user = auth.create_user_with_email_and_password(email, password)
            user_id = user['localId']  # UID from Firebase Auth

            # Store user information in Firestore with user_UID as document ID
            db.collection('users').document(user_id).set({
                'user_name': username,
                'email': email,
                'role': role
            })

            return redirect(url_for('login'))
        except Exception as e:
            print(f"Signup error: {e}")
            return render_template('Registering.html', error=handle_firebase_error(e))

    return render_template('Registering.html')
    
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('user', None)  # Remove the user from the session
    return redirect(url_for('login'))

@app.route('/gallery', methods=['GET', 'POST'])
@login_required
def gallery():
    user_email = session.get('user')
    if not user_email:
        return redirect(url_for('login'))

    try:
        # Fetch user document
        users_ref = db.collection('users')
        user_docs = users_ref.where('email', '==', user_email).limit(1).get()

        if not user_docs:
            return "User not found. Please log in again.", 400

        user_doc = user_docs[0]
        user_uid = user_doc.id
        username = user_doc.to_dict().get('user_name', 'Unknown')

        # Fetch all logs for the user
        logs = fetch_image_logs(user_uid)

        return render_template('Logs.html', username=username, logs=logs)

    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred. Please try again later.", 500


@app.route('/delete_image/<image_uid>', methods=['POST'])
@login_required
def delete_image(image_uid):
    user_email = session.get('user')
    if not user_email:
        return redirect(url_for('login'))

    try:
        # Fetch user document
        users_ref = db.collection('users')
        user_docs = users_ref.where('email', '==', user_email).limit(1).get()

        if not user_docs:
            return "User not found. Please log in again.", 400

        user_doc = user_docs[0]
        user_uid = user_doc.id

        # Delete image metadata from Firestore
        logs_ref = db.collection('users').document(user_uid).collection('logs')
        logs_ref.document(image_uid).delete()

        # Delete image file from Firebase Storage
        firebase_path = f"users/{user_uid}/images/{image_uid}.jpg"
        bucket = storage.bucket()
        blob = bucket.blob(firebase_path)

        if blob.exists():
            blob.delete()

        flash("Image deleted successfully!", "success")
        return redirect(url_for('gallery'))

    except Exception as e:
        print(f"Error occurred: {e}")
        flash("An error occurred while deleting the image.", "error")
        return redirect(url_for('gallery'))

@app.route('/admin', methods=['GET'])
@login_required
@role_required('admin')
def admin_panel():
    # Check if the user is an admin
    user_email = session.get('user')
    user_doc = db.collection('users').where('email', '==', user_email).limit(1).get()[0]
    if user_doc.to_dict().get('role') != 'admin':
        return "Unauthorized Access", 403

    # Fetch all users
    users = [doc.to_dict() | {'id': doc.id} for doc in db.collection('users').stream()]
    return render_template('AdminPanel.html', users=users)

from datetime import datetime

@app.route('/admin/user/<user_id>', methods=['GET'])
@login_required
@role_required('admin')
def admin_user_details(user_id):
    try:
        # Fetch user data
        user_doc = db.collection('users').document(user_id).get()
        if not user_doc.exists:
            return "User not found.", 404

        user_data = user_doc.to_dict()
        user_name = user_data.get('user_name', 'Unknown')

        # Fetch image logs
        image_logs_ref = db.collection('users').document(user_id).collection('logs')
        image_logs = [
            log.to_dict() | {'image_uid': log.id}
            for log in image_logs_ref.stream()
        ]
        for log in image_logs:
            if 'upload_time' in log and log['upload_time']:
                log['upload_time'] = log['upload_time'].strftime('%Y-%m-%d %H:%M:%S')

        # Fetch login logs
        login_logs_ref = db.collection('users').document(user_id).collection('login_logs')
        login_logs = [
            log.to_dict() | {'log_id': log.id}
            for log in login_logs_ref.order_by('timestamp', direction='DESCENDING').stream()
        ]
        for log in login_logs:
            if 'timestamp' in log and log['timestamp']:
                log['timestamp'] = log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

        return render_template(
            'AdminUserDetail.html',
            user_data=user_data,
            image_logs=image_logs,
            login_logs=login_logs
        )

    except Exception as e:
        print(f"Error fetching user details or logs: {e}")
        return "An error occurred. Please try again later.", 500


def fetch_image_logs(user_uid):
    """
    Fetch image logs for a specific user from Firestore.

    Args:
        user_uid (str): The UID of the user.

    Returns:
        list: A list of log entries as dictionaries.
    """
    try:
        # Reference the logs collection for the user
        logs_ref = db.collection('users').document(user_uid).collection('logs')
        
        # Fetch all documents in the logs collection
        logs = []
        for doc in logs_ref.stream():
            log_data = doc.to_dict()
            log_data['image_uid'] = doc.id  # Add document ID as image_uid
            logs.append(log_data)

        # Sort logs by upload time (optional)
        logs.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        return logs
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return []

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
