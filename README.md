# 🧠 AI & ML Interactive Application Suite

Welcome to the **AI & ML Interactive Application**, a multi-section educational tool designed to demonstrate key concepts in machine learning and artificial intelligence through intuitive, hands-on modules.

This project showcases the power and versatility of modern ML techniques, offering users an opportunity to interact with models, visualize outputs, and better understand how algorithms function under the hood. The application is built using **Streamlit** and includes four main sections:

---

## 📊 1. Linear Regression

This section illustrates one of the most fundamental supervised learning algorithms — **Linear Regression**. Users can:

- Explore the relationship between input features and target variables  
- Visualize the fitted regression line in 2D space  
- Adjust parameters and observe the model’s behavior in real time  

## Usage
- Upload CSV file into the appropriate region
- Select the variable which you want to predict
- Select the Predictor variables
- The Mean Square Error (MSE) and the R<sup>2</sup> Score will be displayed with the scatter plot


**Educational Goal:** Understand how predictive models make numerical estimations and how model accuracy is influenced by data patterns.

---

## 🔍 2. Clustering

This module introduces **unsupervised learning** through clustering algorithms like **K-Means**. Users are able to:

- Upload or select datasets to be clustered  
- Observe how the algorithm partitions data based on similarity  
- Visualize clusters and evaluate results

## Usage
- Upload CSV file in the appropriate area
- Select two features to see the 3D cluster plot of the results
- (Optional) Show the center of the clusters


**Educational Goal:** Gain insights into how machines can discover structure in unlabeled data and how grouping can aid in data analysis.

---

## 🧠 3. Neural Networks

A simplified implementation of a **feedforward neural network** allows users to:

- Customize network architecture (epochs, etc.)  
- Train on sample datasets  
- Observe training progress through loss and accuracy plots
- Predict future outputs using the trained model

## Usage
- Upload CSV file into the directed area.
- select target column.
- Select adjustable features and apply binning.
- Indicate test size and adjust hyperparameters.
- Train model and adjust values after training for prediction.


**Educational Goal:** Provide an interactive understanding of deep learning fundamentals and backpropagation mechanics.

---

## 📚 4. Student Handbook Assistant (LLM-Powered)

Leveraging **Large Language Models (LLMs)**, this section introduces a real-world NLP application: a virtual assistant trained to answer questions using a student handbook. Features include:

- Semantic search and document-aware question answering  
- Interaction with an LLM for contextual, human-like responses  
- PDF-based(academic City Student Handbook) knowledge retrieval using vector embeddings  


## Usage
- (Optional) Download the Student Handbook for your own use.
- Ask whatever you would like to know
- Click on "Ask Student Handbook Assistant"
- Wait for you response


**Educational Goal:** Demonstrate how modern language models can enhance information access and serve as intelligent assistants in academic contexts.

---

## 🔧 Technologies Used

- Python  
- Streamlit  
- scikit-learn  
- TensorFlow (for neural networks)  
- LangChain & GEMINIAI (for LLM functionality)  
- PyMuPDFLoader (for document parsing and semantic search)

---
## 🎛 Usage Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Set Up Environment**:
   ```bash
   # Inside your .env file
   GEMINI_API_KEY= "Gemini API Key Here"
   ```
3. **Run Streamlit**:
   ```bash
   streamlit run ML_AI.py
   ```
## (d) Datasets & Models (📚 Student Handbook Assistant (LLM-Powered))

### Datasets
- **Academic City Student Handbook (PDF)**: This text contains the official rules & guidelines for students of Academic City. It clearly defines all information needed by a student of the school in order to live as a contributing member of the ACity community.

### Architecture (📚 Student Handbook Assistant (LLM-Powered))
1. **PDF Parsing & Caching** : Converts the Student Handbook pages to text and stores it for quick retrieval.  
2. **Ranking & Retrieval** : Paragraphs are split and ranked based on shared words with the question asked.  
3. **Generative Model** : **gemini-1.5-flash** merges the context and the user's question to provide answers based on the information in the handbook.

### Methodology (📚 Student Handbook Assistant (LLM-Powered))
- **Retrieval-Augmented Generation (RAG)**: This ranks and inserts top-matching passages into the prompt.  
- **Generative Response**: The AI uses the combined text and query to produce contextually relevant answers.

This application serves as both a learning resource and a demonstration platform for essential ML and AI techniques. It highlights the growing impact of intelligent systems and encourages further exploration into their design and implementation.

**Done by Baldo Giorgio Otu-Quayson**
