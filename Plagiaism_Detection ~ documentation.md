**Project Title: PlagiarismGuard Pro**

---

**1. Project Overview**

The "PlagiarismGuard Pro" is an advanced web application that helps users detect plagiarism in text content. This intelligent system enables users to check text against internet sources or compare two documents to identify similarities and potential plagiarism. Using natural language processing techniques and search capabilities, it efficiently identifies matching content, calculates similarity percentages, and generates comprehensive reports on plagiarized content.

---

**2. Objective**

The primary objectives of this project are to:
- Provide an accessible tool for detecting plagiarism in academic and professional content
- Automate plagiarism detection using web search capabilities and text comparison algorithms
- Generate detailed reports with similarity scores and source identification
- Support various input methods (direct text input and file upload) to enhance user convenience

---

**3. Technologies and Tools**

- **Programming Language:** Python 3.8+
- **Web Framework:** Flask
- **Frontend:** HTML, CSS, and Bootstrap
- **Text Processing:** NLTK (Natural Language Toolkit)
- **Search Capabilities:** Google Search and Bing Search APIs
- **Document Processing:** PyPDF2 for PDF handling
- **Data Manipulation:** Pandas for report generation
- **Text Comparison Algorithms:**
  - Sequence Matcher for similarity calculation
  - KMP (Knuth-Morris-Pratt) algorithm for file-to-file comparison
- **Web Scraping:** BeautifulSoup for extracting text from websites

---

**4. System Requirements**

- **Operating System:** Windows, Linux, or macOS
- **Software Requirements:** Python 3.8 or above
- **Network:** Active internet connection (for web search capabilities)
- **Browser:** Any modern web browser (Chrome, Firefox, Safari, etc.)
- **Tools:** Git for version control, Virtual Environment for dependency isolation

---

**5. Setup Instructions**

**a. Environment Setup**

1. **Clone the Repository:**
   ```
   git clone 
   cd Plagiarism-Detection-Tool
   ```

2. **Create and Activate Virtual Environment:**
   ```
   pip install virtualenv
   virtualenv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

**b. Running the Application**

1. **Start the Flask Application:**
   ```
   python app.py
   ```

2. **Access the Web Interface:**
   - Open a web browser and navigate to `http://localhost:5000`

---

**6. Detailed Project Structure**

```
Plagiarism-Detection-using-ML/
├── app.py                        # Main Flask application file
├── requirements.txt              # Python dependencies
├── word.txt                      # Temporary file for storing text input
├── wsgi.py                       # WSGI entry point for production deployment
├── templates/                    # HTML templates
│   ├── index.html                # Main page template
│   ├── f2f.html                  # File-to-file comparison template
│   └── report.html               # Plagiarism report template
├── static/                       # Static assets (CSS, JS, images)
├── uploads/                      # Directory for uploaded files (created at runtime)
├── venv/                         # Virtual environment (created during setup)
└── __pycache__/                  # Python cache files
```

---

**7. Core Components**

- **Web Interface Module:**  
  Provides an intuitive UI for users to input text or upload files, view plagiarism detection results, and generate reports.

- **Text Processing Engine:**  
  Uses NLTK to tokenize text, remove stop words, and prepare content for comparison. This ensures efficient and accurate plagiarism detection.

- **Web Search Integration:**  
  Leverages Google and Bing search capabilities to find potential sources of plagiarized content across the internet.

- **Text Comparison Engine:**  
  - **Web Verification:** Compares input text against web sources to identify plagiarism
  - **File-to-File Comparison:** Uses KMP algorithm for comparing two documents
  - **Similarity Calculation:** Quantifies the degree of similarity between texts

- **Report Generation:**  
  Creates detailed reports showing plagiarized content, source URLs, and similarity percentages using Pandas for data formatting.

---

**8. Usage Guide**

**a. Web-based Plagiarism Detection:**

1. **Enter Text or Upload File:**
   - Type or paste text in the provided text area, or
   - Upload a compatible document file (.pdf, .doc, .docx, .txt)

2. **Initiate Plagiarism Check:**
   - Click "Check Plagiarism" to compare the content against internet sources

3. **View Results:**
   - See the overall plagiarism percentage
   - Review highlighted plagiarized content with source links

4. **Generate Report:**
   - Click "Show Report" to create a detailed plagiarism report
   - The report includes similarity scores for each matching source and total similarity percentage

**b. File-to-File Comparison:**

1. **Navigate to Compare F2F:**
   - Click the "Compare F2F" link in the navigation bar

2. **Upload Two Files:**
   - Upload two documents you want to compare for similarities

3. **Initiate Comparison:**
   - Click "Check Plagiarism" to analyze the files using the KMP algorithm

4. **View Results:**
   - See the similarity percentage between the two files
   - Review matching content highlighted in the results

---

**9. Key Features**

- **Dual Detection Methods:**
  - Web-based plagiarism detection using search engines
  - File-to-file comparison for direct document analysis

- **Multiple Input Options:**
  - Direct text input
  - File upload support for various document formats

- **Comprehensive Reporting:**
  - Percentage-based similarity scores
  - Source identification with links
  - Detailed breakdown of matching content

- **Advanced Text Processing:**
  - Stop word removal for improved accuracy
  - Sentence tokenization for contextual comparison
  - Efficient search query generation

---

**10. Architecture Diagram**

<div style="page-break-inside: avoid;">
  <img src="Architecture.png" alt="Architecture Diagram" style="width:70%; height:50%; display: block; margin: auto;">
</div>

- **Layered Architecture:** The diagram illustrates a layered architecture for a Plagiarism Detection System, clearly separating responsibilities into the User Interface, Application, Service, Data, and External Services layers. This separation promotes modularity, maintainability, and scalability.

- **User Interface Layer:** This layer consists of a web browser interfacing with HTML templates that incorporate Bootstrap components for responsive UI. It serves as the primary point of interaction between users and the system.

- **Application Layer:** Built using Flask, this layer manages the core application logic. The `Main App` initializes the application, and `Route Handlers` manage routing and control the flow of data between the front end and backend services.

- **Service Layer:** This is the core logic layer, responsible for various functionalities such as:

  - **Text Processing:** Cleans and prepares the input for further analysis.
  - **Web Search Service:** Queries external search APIs to retrieve potentially plagiarized content.
  - **Similarity Engine:** Compares user content with retrieved data to detect similarities.
  - **Report Generator:** Compiles findings into detailed plagiarism reports.
  - **File Comparison (KMP):** Utilizes the Knuth-Morris-Pratt algorithm for efficient file content comparison.

- **Data Layer:** Manages local data storage, including:

  - **File Storage:** Stores user-submitted files.
  - **Temporary Text Storage:** Holds intermediate processed text data used during comparisons.

- **External Integration:** The system connects with external APIs such as Google and Bing Search to fetch real-world web content. This integration enhances the engine's ability to detect copied material from online sources.

- **Feedback Flow:** The diagram depicts inter-service communication flows, especially between the Similarity Engine, Report Generator, and Web Content sources, ensuring accurate and comprehensive plagiarism evaluation.

---
**11. Future Enhancements**

- **Machine Learning Integration:** Implement ML algorithms for more accurate plagiarism detection
- **Support for Additional Languages:** Extend beyond English to support multiple languages
- **API Development:** Create an API for integrating plagiarism detection capabilities with other applications
- **Enhanced Visualization:** Implement better visual representation of plagiarized content
- **Bulk Processing:** Add support for checking multiple files simultaneously
---