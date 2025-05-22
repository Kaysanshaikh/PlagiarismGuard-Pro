# PlagiarismGuard Pro

## Overview
A comprehensive plagiarism detection tool that offers two main functionalities:
1. Web-based text similarity checking against internet sources
2. File-to-file plagiarism detection using KMP (Knuth-Morris-Pratt) algorithm

## Features

### Web Content Similarity Detection
- Tokenizes input text into sentences and removes stopwords using NLTK for optimized search queries
- Utilizes both Google and Bing search engines to maximize detection coverage
- Extracts text from web sources using BeautifulSoup and calculates similarity with SequenceMatcher
- Filters out non-academic sources like social media and e-commerce sites
- Identifies and highlights matching content from credible web sources
- Calculates percentage-based similarity scores for individual matches and overall content
- Detailed report generation showing:
  - Similar words found online
  - Source websites
  - Percentage of content copied from each source

### File Comparison
- Upload and compare two documents for similarity
- KMP algorithm implementation for efficient string matching
- Detailed similarity percentage calculation

## Technical Snapshot
- HTML, CSS, Bootstrap for frontend
- Flask for backend
- Python backend for text processing
- googlesearch package for web queries
- Custom implementation of KMP algorithm

## Installation and Setup
```bash
# Clone the repository
git clone 

# Navigate to project directory
cd Plagiarism-Detection-Tool

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Usage
### Web Content Check
1. Enter or paste text in the input box
2. Click "Check Plagiarism"
3. View highlighted similar content
4. Check similarity percentage
5. Generate detailed report

### File Comparison
1. Upload two files using the file selection interface
2. Click "Compare Files"
3. View similarity percentage and matching content

## Future Enhancements
- PDF and document format support
- API integration for third-party applications
- Improved accuracy through machine learning
- Bulk file processing

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.