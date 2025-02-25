from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import io
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# In-memory storage for users (for demonstration only)
users = {}

@app.route('/')
def home():
    return jsonify({"message": "Welcome to ItCapstone Resume and Job Description Comparison Tool!"})

# Registration Endpoint
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('full_name')

    # Validate input
    if not email or not password or not full_name:
        return jsonify({"error": "Email, password, and full name are required."}), 400

    # Check if user already exists
    if email in users:
        return jsonify({"error": "User already exists."}), 400

    # Hash the password and store user
    hashed_password = generate_password_hash(password)
    users[email] = {"full_name": full_name, "password": hashed_password}
    return jsonify({"message": "User registered successfully."}), 201

# Login Endpoint
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = users.get(email)
    # Validate user existence and password
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials."}), 401

    return jsonify({"message": "Login successful."}), 200

# Helper function: Extract text from PDF files
def extract_text_from_pdf(file_stream):
    reader = PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Helper function: Compute similarity using TF-IDF and cosine similarity
def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return cosine_sim[0][0]

# File Upload Endpoint for Resume and Job Description Comparison
@app.route('/upload', methods=['POST'])
def upload():
    # Ensure both files are provided
    if 'resume' not in request.files or 'job_description' not in request.files:
        return jsonify({"error": "Both resume and job description files are required."}), 400

    resume_file = request.files['resume']
    jd_file = request.files['job_description']

    # Read and extract text from the uploaded PDF files
    resume_text = extract_text_from_pdf(io.BytesIO(resume_file.read()))
    jd_text = extract_text_from_pdf(io.BytesIO(jd_file.read()))

    # Compute the similarity score
    similarity_score = compute_similarity(resume_text, jd_text)
    return jsonify({"similarity_score": similarity_score}), 200

# A simple test endpoint
@app.route('/test', methods=['GET'])
def test_route():
    return jsonify({"message": "Test route is working!"})

if __name__ == "__main__":
    app.run(debug=True)
