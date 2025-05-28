import streamlit as st

st.set_page_config(
        page_title="Machine Learning & Artificial Intelligence Consult",
    layout="wide",
    initial_sidebar_state="expanded"
)



##Home page
def home():
    st.title("Machine Learning & Artificial Intelligence Consult")

    st.subheader("🏠 Welcome to Your Machine Learning & Artificial Intelligence Virtual Assistant!")
    st.write(" Explore a powerful and user-friendly platform designed to help you understand, analyze, and interact with data like never before. Our app combines classic machine learning tools with modern AI to deliver an all-in-one educational and analytical experience.")

    st.subheader("🔧 App Features:")
    st.subheader("📈 Linear Regression Mode")
    st.write("Visualize relationships and make predictions using simple yet effective linear models. Great for understanding trends and estimating future outcomes from your data.")

    st.subheader("🔍 Clustering Mode")
    st.write("Discover hidden patterns by grouping similar data points using K-Means Clustering. A perfect tool for exploring natural segments within your dataset.")

    st.subheader(" 🧠 Neural Network Training Mode")
    st.write("Dive into the basics of deep learning with a streamlined MLP (Multi-Layer Perceptron) trainer. Train simple neural networks and see how machines learn from data.")

    st.subheader(" 📚 Student Handbook Assistant (LLM-Powered)")
    st.write(" Need quick answers from your student handbook? Use our built-in AI assstant powered by a Large Language Model to get instant, accurate responses from the Student Handbook — like having a personal tutor at your fingertips.")




##Regression Section
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def linear_regression_section():
    st.title("Linear Regression App")

    st.subheader("📈 Linear Regression Predictor")
    st.write("This section uses a simple linear regression model to analyze the relationship between input variables and predict future outcomes.")
    st.write("This section requires that you upload a file (CSV). The model then uses this information to provide you with the select data based on various metrics.")
    st.write("Displays a Scatter plot of the Actual data vs the Predicted values.")
    st.write("Whether you're exploring trends or estimating values, this tool provides a quick and intuitive way to model linear patterns in your data.")



    
    file = st.file_uploader("Upload your CSV file", type=["csv"])

    if file:
        data= pd.read_csv(file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

     
        columns = data.columns.tolist()
        target = st.selectbox("Select the target variable (what would you like to predict?: )", columns)
        predictors = st.multiselect("Select predictor variable(s)", [col for col in columns if col != target])

        if predictors and target:
            x = data[predictors]
            y = data[target]

    
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

     
            model = LinearRegression()
            model.fit(x_train, y_train)

    
            y_predict = model.predict(x_test)
            MSE = mean_squared_error(y_test, y_predict)
            act_price = list(y_test)
            pred_price = list(y_predict)
            st.subheader("Model Performance")
            st.write(f"**R² Score:** {r2_score(y_test, y_predict):.4f}")
            st.write(f"**Mean Error (MSE):** {MSE}")


            # Plot
            st.subheader("Actual vs Predicted")
            fig, axis = plt.subplots()
            sns.scatterplot(x = range(100), y = act_price[:100], color = 'red', label = 'Actual Prices')
            sns.scatterplot(x = range(100), y = pred_price[:100], color = 'blue', label = 'Predicted Prices')
            axis.set_xlabel("Actual Values")
            axis.set_ylabel("Predicted Values")
            axis.set_title("Actual vs Predicted")
            st.pyplot(fig)
        else:
            st.warning("Please select both target and predictor variables.")
    else:
        st.info("Awaiting CSV file upload.")





##Clustering Section
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def Clustering_Section():
    st.title("Clustering App")
    st.subheader("🔍Clustering Visualizer (K-Means)")
    st.write("This section applies K-Means clustering to group your data into distinct clusters based on similarity."
    "It is an unsupervised machine learning technique that helps reveal hidden patterns, segment data, and explore natural groupings — all without needing labeled input.")

    

    file = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if file:
        data = pd.read_csv(file)
        st.write("Preview of Dataset:")
        st.dataframe(data.head())

        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()

        features = st.multiselect("Select features for clustering", numeric_cols)

        if len(features) >= 2:
            x = data[features]

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(x)

            # Number of clusters
            n_clusters = st.slider("Select number of clusters (K)", 2, 10, 3)

            # Run KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(X_scaled)

            data["Cluster"] = labels

            # Visualization
            st.subheader("Cluster Visualization")

            if len(features) == 2:
                fig, axis = plt.subplots()
                sns.scatterplot(
                    x=x[features[0]],
                    y=x[features[1]],
                    hue=labels,
                    palette="Set2",
                    ax=axis
                )
                axis.set_title("2D Cluster Plot")
                st.pyplot(fig)

            elif len(features) >= 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(
                    x[features[0]],
                    x[features[1]],
                    x[features[2]],
                    c=labels,
                    cmap="Set2",
                    s=60
                )
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_zlabel(features[2])
                ax.set_title("3D Cluster Plot")
                st.pyplot(fig)

            # Show cluster centers if needed
            if st.checkbox("Show Cluster Centers (scaled)"):
                st.write(pd.DataFrame(kmeans.cluster_centers_, columns=features))

        else:
            st.warning("Please select at least two numeric features for clustering.")
    else:
        st.info("Upload a CSV file to get started.")




##Neural Network Section
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

def Neural_Network_Section():
    st.subheader("🧠Neural Network Section")
    st.header("Neural Network Builder")
    st.write("This section allows users to build, train, and evaluate neural network models.")
    st.write("It provides tools for configuring layers, activation functions, learning rates, and other hyperparameters. Visualizations help track training progress, performance metrics, and predictions.")
    st.write("Ideal for tasks like classification, regression, and deep learning experiments.")

    file = st.file_uploader("Upload CSV", type=["csv"], key="neural_network")
    if file:
        data = pd.read_csv(file)
        st.dataframe(data.head())
        columns = data.columns.tolist()
        target_column = st.selectbox("Select target column", columns)
        select_columns = st.multiselect(
            "Select features", options = [col for col in columns if col != target_column],
            help = "Press enter to confrim your selection."
            )

        if not select_columns:
            st.warning("Please select at least one feature column")
            st.stop()
        if st.markdown("drop rows with missing values"):
            data = data.dropna(subset=select_columns +[target_column])
            st.success("Empty rows have been removed")

        y = data[target_column].values
        diff_target_vals = data[target_column].nunique()
        
        if diff_target_vals > 50:
            st.warning(f"Target column '{target_column}' has {diff_target_vals}' unique values.")
            if st.checkbox("Apply binning to target"):
                data[target_column] = pd.qcut(data[target_column], q = 10, labels = False)
                st.info("Target has been binned.")
                y = data[target_column].values
            else:
                st.error("Target is continuous. Apply binning to continue...")
                st.stop()

        if not np.issubdtype(y.dtype, np.number):
            label_encoder = LabelEncoder()
            y= label_encoder.fit_transform(y)

        else:
            label_encoder = None

        unique_class = np.unique(y)
        num_class = len(unique_class)
        if not np.array_equal(np.sort(unique_class), np.arange(num_class)):
            st.error(f"Class label must be from 0 - {num_class - 1}.")
            st.stop()

        num_feat = [col for col in select_columns if pd.api.types.is_numeric_dtype(data[col])]
        cat_feat = [col for col in select_columns if not pd.api.types.is_numeric_dtype(data[col])]
        st.write("**Numeric features:**", num_feat)
        st.write("**Categorical features:**", cat_feat)

        if num_feat:
            scaler = StandardScaler()
            x = scaler.fit_transform(data[num_feat].astype(float))

        else:
            x = np.empty((data.shape[0], 0))


        if cat_feat:
            x_cat_df = pd.get_dummies(data[cat_feat], drop_first=False)
            x_cat = x_cat_df.values
            cat_columns = x_cat_df.columns.tolist()

        else:
            x_cat = np.empty((data.shape[0], 0))
            cat_columns = []

        x = np.hstack([x, x_cat])
        st.write("Final matrix shape after processing: ", x.shape)

        test_size = st.slider("Test set size", 10, 20, 20) / 100
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)
        st.subheader("Hyperparameter Configuration")
        epochs = st.slider("Epochs", 1, 100, 10)
        batch_size = st.slider("Batch Size", 8, 128, 16)
        learn_rate = st.number_input("Learning Rate", 0.001, 1.0, 0.001, 0.001, "%.4f")


        if num_class ==2:
                final_activation = 'sigmoid'
                loss_func ='binary_crossentropy'

        else: 
                final_activation = 'softmax'
                loss_func = 'sparse_categorical_crossentropy'

        if st.button("Train Model"):
            model = tf.keras.Sequential ([
                tf.keras.layers.Dense(32, activation = 'relu', input_shape = (x_train.shape[1],)),
                tf.keras.layers.Dense(num_class, activation = final_activation)
            ])

            model.compile(             
                optimizer = tf.keras.optimizers.Adam( learning_rate = learn_rate, name = 'adam'),
                loss = loss_func,
                metrics = ['accuracy']
            )

            st.spinner("Training in progress...")
            progress_bar = st.progress(0)


            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_bar.progress((epoch + 1) / epochs)

            history = model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[StreamlitCallback()]
            )
            st.session_state.nn_trained = True
            st.session_state.nn_model = model
            st.session_state.nn_history = history
            st.session_state.nn_scaler = scaler
            st.session_state.nn_label_encoder = label_encoder
            st.session_state.nn_numeric_features = num_feat
            st.session_state.nn_cat_columns = cat_columns
            st.session_state.nn_data = data  # store original DataFrame for prediction alignment
            st.success("Model trained successfully!")

            st.write("Enter custom values for each original feature to get a prediction:")
        
            custom_numeric = {}
            custom_categorical = {}
            num_cols_ui = 3
            cols_ui = st.columns(num_cols_ui)
        
            for i, feat in enumerate(st.session_state.nn_numeric_features):
                with cols_ui[i % num_cols_ui]:
                    default_val = float(st.session_state.nn_data[feat].mean())
                    custom_numeric[feat] = st.number_input(f"Enter value for {feat}", value=default_val)
        
            for i, feat in enumerate(data.columns.difference(st.session_state.nn_numeric_features + [target_column])):
        
                if feat in st.session_state.nn_cat_columns:
                    with cols_ui[i % num_cols_ui]:
                        default_val = st.session_state.nn_df[feat].mode()[0] if not st.session_state.nn_df[feat].mode().empty else ""
                        custom_categorical[feat] = st.text_input(f"Enter value for {feat}", value=str(default_val))
            

            st.subheader("Make Prediction")
            inputs = {col: st.number_input(f"Enter {col}", value=float(data[col].mean())) for col in select_columns}

        if st.button("Predict"):
            try:
                if st.session_state.nn_cat_columns:
                    custom_num_df = pd.DataFrame([custom_numeric])
                    custom_num_scaled = st.session_state.nn_scaler.transform(custom_num_df.astype(float))

                else:
                    custom_num_scaled = np.empty((1, 0))

                if st.session_state.nn_cat_columns:
                    custom_cat_df = pd.DataFrame([custom_categorical])
                    custom_cat_dum = pd.get_dummies(custom_cat_df, drop_first = False)
                    custom_cat_al = custom_cat_dum.reindex(columns = st.session_state.nn_cat_columns, fill_val = 0).values

                else:
                    custom_cat_al = np.empty((1, 0))
                full_inp = np.hstack
                prediction = st.session_state.nn_model.predict(full_inp)
                predicted_index = np.argmax(prediction, axis=1)[0]
                if st.session_state.nn_le is not None:
                    predicted_class = st.session_state.nn_le.inverse_transform([predicted_index])[0]
                else:
                    predicted_class = predicted_index
                st.success(f"Predicted {target_column}: {predicted_class}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                


##LLM Section
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
import google.generativeai as genai
import re
import base64

def LLM():
    st.title("Student Handbook Assistant")
    st.subheader(" 📚Academic City Handbook Assistant")
    st.write("This section uses a large language model to help users explore and understand the school's handbook. You can ask questions in natural language—like policies on attendance, grading, or discipline—and get clear, accurate answers pulled directly from the official document.")
    st.write("It is a smart, searchable guide designed to make the handbook easier and faster to navigate.")

    load_dotenv()
    api_key = os.getenv("GEMINIAI_API_KEY")
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel("gemini-2.0-flash")


    st.markdown("Ask anything you would like to know")


    def read_text(data_path):
        try:
            with open(data_path, "rb") as file:
                reader = PyMuPDFLoader(data_path)
                docs = reader.load()

                return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            st.error(f"Error loading Handbook: {str(e)}")
            return ""
        
    def get_pdf_base64(data_path):
        try:
            with open(data_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            st.error(f"Error encoding PDF: {e}")
            return ""    

            
    def pdf_download(data_path):

        if st.button("Download PDF"):
            base64_pdf = get_pdf_base64(data_path)
            file = os.path.basename(data_path)
            st.session_state.pdf_ready = True
            st.session_state.base64_pdf = base64_pdf
            st.session_state.file_name = file
            st.rerun()
    if 'pdf_ready' in st.session_state and st.session_state.pdf_ready:
        if st.download_button("Download PDF", data=st.session_state.base64_pdf, 
                            file_name=st.session_state.file_name, mime="application/pdf"):
            st.info("Download the PDF and view it with your local PDF reader or browser.")
            st.session_state.pdf_ready = False


    data_source={"Academic City Student Handbook (PDF)": "handbook.pdf"}

    data_set = st.selectbox("Load Handbook", list(data_source))
    select_path = data_source[data_set]
    content = ""

    if select_path.endswith (".pdf"):

        with st.spinner("Processing PDF..."):
            content = read_text(select_path)
        
        with st.expander("Download Options"):
                pdf_download(select_path)
    else: 
        st.error("Unsupported File Format")

    paragraphs = content.split("\n\n")
    num_pages = 5
    question= st.text_input("What would you like to know from the Handbook? ")

    if question:
        question_con = set(re.findall(r'\w+', question.lower()))
        rank = []

        for par in paragraphs:
            p_con = set(re.findall(r'\w+', par.lower()))
            score = len(question_con.intersection(p_con))
            rank.append((score, par))
        rank.sort(key = lambda x: x[0], reverse = True)

        max_pages = min(num_pages, len(rank))
        retrieve_context = "\n\n".join([par for score, par in rank[:max_pages]])

    else:
        retrieve_context = content

    if st.button("Ask Handbook Assistant"):
        if retrieve_context and question:
            with st.spinner("Retreiving Response"):
                try:
                    prompt = f"From the handbook: \n{retrieve_context}\n\nYou asked {question}\n\n According to the handbook:"

                    model = genai.GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    st.success("Here You Go:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Response not found: {str(e)}")
                    st.write("Question is not in the handbook. Please try again")
        else:
            st.warning("Please ask a question")


st.sidebar.title("Options")
st.sidebar.write("In this area, you can choose from the various Machine learning & Artificial Intelligence tools at your disposal")
section = st.sidebar.selectbox(
        "Choose a section:",
        ("🏠Home", "📈Linear Regression", "🔍Clustering", "🧠Neural Network", "📚 Student Handbook Assistant")
    )

if section == "📈Linear Regression":
    linear_regression_section()
elif section == "🔍Clustering":
    Clustering_Section()
elif section == "🧠Neural Network":
    Neural_Network_Section()
elif section == "🏠Home":
    home()
elif section == "📚 Student Handbook Assistant":
    LLM()