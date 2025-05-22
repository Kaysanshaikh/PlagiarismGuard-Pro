from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup as bs
from difflib import SequenceMatcher
from googlesearch import search
import pandas as pd
import requests
import warnings
import PyPDF2
import nltk
import re
import os
import io
import docx
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", module='bs4')

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Set up upload folder and allowed extensions
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Create a Flask app instance and configure it
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.add_url_rule("/uploads/<n>", endpoint="download_file", build_only=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['SECRET_KEY'] = 'super secret key'

# Setup requests with retry capability
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Get stopwords once for efficiency
stop_words = set(nltk.corpus.stopwords.words('english'))

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file"""
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_text_from_file(file_path):
    """Extract text from a file based on its extension"""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        return ""

def searchBing(query, num):
    """Search Bing for a given query and return a list of URLs"""
    urls1 = []
    url1 = 'https://www.bing.com/search?q=' + query
    
    try:
        page = session.get(url1, headers={'User-agent': 'Mozilla/5.0'}, timeout=10)
        soup = bs(page.text, 'html.parser')
        
        for link in soup.find_all('a'):
            url1 = str(link.get('href'))
            if url1.startswith('http'):
                if not url1.startswith('https://go.m') and not url1.startswith('https://maps.'):
                    urls1.append(url1)
        
        return urls1[:num]
    except Exception as e:
        print(f"Error in Bing search: {e}")
        return []

def searchGoogle(query, num):
    """Search Google for a given query and return a list of URLs"""
    urls2 = []
    url2 = 'https://www.google.com/search?q=' + query
    
    try:
        page = session.get(url2, headers={'User-agent': 'Mozilla/5.0'}, timeout=10)
        soup = bs(page.text, 'html.parser')
        
        for link in soup.find_all('a'):
            url2 = str(link.get('href'))
            if url2.startswith('http'):
                if not url2.startswith('https://go.m') and not url2.startswith('https://maps.google'):
                    urls2.append(url2)
        
        return urls2[:num]
    except Exception as e:
        print(f"Error in Google search: {e}")
        return []

def extractText(url):
    """Extract text content from a URL"""
    try:
        page = session.get(url, headers={'User-agent': 'Mozilla/5.0'}, timeout=10)
        soup = bs(page.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        return ""

def purifyText(string):
    """Remove stopwords from a string"""
    words = nltk.word_tokenize(string)
    return (" ".join([word for word in words if word not in stop_words]))

def webVerify(string, results_per_sentence=2):
    """Verify text against web sources and return matching sites"""
    sentences = nltk.sent_tokenize(string)
    matching_sites = []
    
    # Search for the entire string
    for url in searchBing(query=string, num=results_per_sentence):
        matching_sites.append(url)
    for url in searchGoogle(query=string, num=results_per_sentence):
        matching_sites.append(url)
    
    # Search for each sentence (limit to first 5 sentences for performance)
    for sentence in sentences[:5]:
        if len(sentence.split()) > 5:  # Only search meaningful sentences
            for url in searchBing(query=sentence, num=results_per_sentence):
                matching_sites.append(url)
            for url in searchGoogle(query=sentence, num=results_per_sentence):
                matching_sites.append(url)
    
    # Return unique matching sites
    return list(set(matching_sites))

def similarity(str1, str2):
    """Calculate similarity ratio between two strings"""
    return (SequenceMatcher(None, str1, str2).ratio()) * 100

def generate_pie_chart(original_percentage, plagiarism_percentage):
    """Generate a pie chart showing original vs plagiarized content percentages"""
    labels = ['Original', 'Plagiarized']
    sizes = [original_percentage, plagiarism_percentage]
    colors = ['#3498db', '#e74c3c']
    explode = (0, 0.1)  # explode the 2nd slice (plagiarized)
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Save the chart to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    
    return f'data:image/png;base64,{image_base64}'

def get_report(text):
    """Generate a plagiarism report for the input text"""
    matching_sites = webVerify(purifyText(text))
    matches = {}
    
    # Calculate similarity scores
    total_similarity = 0
    sites_checked = 0
    
    for i, site in enumerate(matching_sites[:10]):  # Limit to first 10 sites for performance
        try:
            site_text = extractText(site)
            if site_text:
                similarity_score = similarity(text, site_text)
                matches[site] = similarity_score
                total_similarity += similarity_score
                sites_checked += 1
        except Exception as e:
            print(f"Error processing site {site}: {e}")
    
    # Calculate average similarity
    avg_similarity = total_similarity / sites_checked if sites_checked > 0 else 0
    
    # Cap similarity at 100%
    plagiarism_percentage = min(avg_similarity, 100)
    original_percentage = 100 - plagiarism_percentage
    
    # Generate pie chart
    chart_image = generate_pie_chart(original_percentage, plagiarism_percentage)
    
    # Prepare results
    result = {
        'matches': matches,
        'plagiarism_percentage': plagiarism_percentage,
        'original_percentage': original_percentage,
        'chart_image': chart_image
    }
    
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle the main page with text input or file upload"""
    if request.method == 'POST':
        # Check if text is provided
        if request.form.get('text', '').strip():
            text = request.form['text']
            return redirect(url_for('plagiarism_check', text=text))
        
        # Check if file is provided
        elif 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            
            if not allowed_file(file.filename):
                flash('File type not allowed. Please upload PDF, DOCX, or TXT files only.')
                return redirect(request.url)
                
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            return redirect(url_for('plagiarism_check', filename=filename))
        
        else:
            flash('Please provide either text or a file to check for plagiarism.')
            return redirect(request.url)
            
    return render_template("index.html")

@app.route('/plagiarism-check', methods=['GET'])
def plagiarism_check():
    """Process the text input or file upload and check for plagiarism"""
    text = request.args.get('text')
    filename = request.args.get('filename')
    
    if not text and not filename:
        flash('No input provided')
        return redirect(url_for('index'))
    
    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            flash('File not found')
            return redirect(url_for('index'))
        
        text = extract_text_from_file(file_path)
    
    # Get plagiarism report
    report_data = get_report(text)
    
    # Show selected matches
    matches_df = pd.DataFrame(
        {'Similarity (%)': {k: f"{v:.2f}%" for k, v in report_data['matches'].items()}}
    )
    
    # Create HTML table from DataFrame
    matches_table = matches_df.to_html(classes="table table-striped")
    
    return render_template(
        "report.html", 
        plagiarism_percentage=round(report_data['plagiarism_percentage'], 2),
        original_percentage=round(report_data['original_percentage'], 2),
        chart_image=report_data['chart_image'],
        matches_table=matches_table,
        input_text=text[:500] + "..." if len(text) > 500 else text
    )

@app.route('/api/check-plagiarism', methods=['POST'])
def api_check_plagiarism():
    """API endpoint for checking plagiarism"""
    text = request.form.get('text')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Get plagiarism report
    report_data = get_report(text)
    
    return jsonify({
        'plagiarism_percentage': round(report_data['plagiarism_percentage'], 2),
        'original_percentage': round(report_data['original_percentage'], 2),
        'matches': {k: round(v, 2) for k, v in report_data['matches'].items()}
    })

@app.route('/compare', methods=['GET', 'POST'])
def compare_documents():
    """Compare two documents for similarity"""
    if request.method == 'POST':
        # Get text inputs if provided
        text1 = request.form.get('text1', '').strip()
        text2 = request.form.get('text2', '').strip()
        
        # Check for file uploads
        file1 = None
        file2 = None
        
        if 'file1' in request.files and request.files['file1'].filename:
            file1 = request.files['file1']
            if not allowed_file(file1.filename):
                flash('File type not allowed for Document 1. Please upload PDF, DOCX, or TXT files only.')
                return redirect(request.url)
                
            filename1 = secure_filename(file1.filename)
            file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            file1.save(file_path1)
            text1 = extract_text_from_file(file_path1)
        
        if 'file2' in request.files and request.files['file2'].filename:
            file2 = request.files['file2']
            if not allowed_file(file2.filename):
                flash('File type not allowed for Document 2. Please upload PDF, DOCX, or TXT files only.')
                return redirect(request.url)
                
            filename2 = secure_filename(file2.filename)
            file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            file2.save(file_path2)
            text2 = extract_text_from_file(file_path2)
        
        # Ensure both documents are provided
        if not text1 or not text2:
            flash('Please provide both documents for comparison.')
            return redirect(request.url)
        
        # Calculate similarity
        similarity_score = similarity(text1, text2)
        
        # Count words in each document
        doc1_words = text1.split()
        doc2_words = text2.split()
        doc1_word_count = len(doc1_words)
        doc2_word_count = len(doc2_words)
        
        # Calculate matching words (approximate)
        matching_words = int((similarity_score / 100) * min(doc1_word_count, doc2_word_count))
        
        # Highlight matching content (basic implementation)
        # For a more sophisticated implementation, you would use difflib or similar
        doc1_content = text1[:1000] + "..." if len(text1) > 1000 else text1
        doc2_content = text2[:1000] + "..." if len(text2) > 1000 else text2
        
        return render_template(
            'compare.html',
            similarity_score=similarity_score,
            doc1_content=doc1_content,
            doc2_content=doc2_content,
            doc1_word_count=doc1_word_count,
            doc2_word_count=doc2_word_count,
            matching_words=matching_words
        )
        
    return render_template('compare.html')

if __name__ == '__main__':
    app.run(debug=True)
