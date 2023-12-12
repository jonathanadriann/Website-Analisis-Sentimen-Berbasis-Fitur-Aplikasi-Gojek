import streamlit as st
import pandas as pd
import numpy as np
import base64
import string, re
import clf
import matplotlib.pyplot as plt
import string
import nltk
import torch
import joblib
import os

from PIL import Image
from streamlit_option_menu import option_menu
from google_play_scraper import app
from google_play_scraper import Sort, reviews
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from googletrans import Translator
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score, accuracy_score

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

if 'df' not in st.session_state:
    st.session_state['df'] = None

def translate_to_bahasa(text):
    try:
        translator = Translator()
        translated = translator.translate(text, src='en', dest='id')
        return translated.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # Return the original text if translation fails

def cleansing(data):
    # Check if the data is a string
    if isinstance(data, str):
        # Define a list of words to keep
        words_to_keep = ["tidak", "perlu", "penting"]  # Add the words you want to keep

        # Remove punctuation
        remove = string.punctuation
        translator = str.maketrans(remove, ' ' * len(remove))
        data = data.translate(translator)

        # Remove ASCII and Unicode
        data = data.encode('ascii', 'ignore').decode('utf-8')
        data = re.sub(r'[^\x00-\x7f]', r'', data)

        # Translate English to Bahasa Indonesia
        data = translate_to_bahasa(data)

        # Tokenization
        words = nltk.word_tokenize(data)

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word, wordnet.VERB) for word in words]

        # Lowercase
        data = ' '.join(lemmatized_words).lower()

        # Remove stopwords and keep specific words
        stopwords_id = set(stopwords.words('indonesian'))
        data = ' '.join([word for word in data.split() if word not in stopwords_id or word in words_to_keep])

        return data
    else:
        # If data is not a string, return an empty string or handle it as needed
        return ""

@st.cache_data()
def load_data(df):
    df['content'] = df['content'].apply(cleansing)

    #Drop column 'at', 'userName'
    df = df.drop(columns=['at', 'userName'])
    return df

# horizontal menu
selected = option_menu(
    menu_title = "Hello, User !",  # Add a menu title
    options = ["Home", "Analysis", "About Me"],
    icons = ["house", "bar-chart" ,"people"],
    menu_icon = "person",
    default_index = 0,
    orientation = "horizontal"
    )

if selected == "Home":
    text1 = """<div style='text-align: justify;'> Dalam era digital yang terus berkembang dari waktu ke waktu, teknologi informasi dan komunikasi telah mengubah berbagai macam aspek kehidupan manusia. Dimulai dari cara manusia berinteraksi, bertransaksi, dan mengakses berbagai layanan yang ada. Aplikasi berbasis platform telah menjadi salah satu contohnya, dimana manusia semakin terbiasa akan pola baru dalam kehidupan sehari-hari, seperti hadirnya aplikasi layanan transportasi dan makanan berbasis teknologi, yaitu Gojek, yang telah membawa banyak dampak signifikan dalam cara masyarakat bergerak, berbelanja, dan beraktivitas.</div>"""
    
    text2 = """<div style='text-align: justify;'> \n Berdasarkan gambar diatas, Gojek tercatat sebagai layanan ojek online favorit masyarakat Indonesia. Tercatat ada 82% responden yang menggunakan layanan milik PT GoTo Gojek Tokopedia Tbk. tersebut, meski memiliki aplikasi lainnya. Sebagai platform aplikasi berbasis online, Gojek telah memperkenalkan konsep layanan serba ada melalui platformnya yang terintegrasi. 
\n Tidak hanya menyediakan layanan transportasi, Gojek juga menyediakan berbagai layanan lain seperti pengiriman makanan, pembayaran tagihan, dan bahkan pembelian produk. Dengan demikian, Gojek bergerak dalam berbagai sektor, menggabungkan teknologi dan layanan tradisional untuk memberikan solusi dalam kehidupan sehari-hari masyarakat. Oleh sebab itu, ulasan pelanggan yang diterima sangat penting adanya. Dimana, ulasan yang sangat beragam dari pengguna inilah yang mencerminkan pengalaman mereka dalam menggunakan layanan-layanan tersebut."""

    text3 = """\n Google Play Store sendiri merupakan platform distribusi digital yang dikembangkan oleh Google. Platform ini merupakan toko resmi tempat pengguna dapat mengunduh dan memperbarui berbagai jenis aplikasi dan permainan untuk perangkat mereka. Pengguna dapat mencari berbagai aplikasi yang tersedia, termasuk aplikasi Gojek.
\n Google Play Store juga menyediakan informasi penting tentang aplikasi, seperti deskripsi, ulasan pengguna, peringkat, dan sebagainya yang membantu pengguna dalam membuat keputusan sebelum mengunduh aplikasi tertentu. Dengan demikian, Google Play Store berperan sebagai gerbang utama bagi pengguna aplikasi di dalamnya untuk mengakses berbagai aplikasi, serta memungkinkan mereka untuk memberikan komentar atau keluhan terhadap aplikasi tersebut dengan mudah melalui perangkat mereka. 
\n Pada kalimat komentar seperti, “Apk gojek sangat membantu aktivitas saya” dapat dikatakan sebagai komentar yang baik, karena memiliki makna sentimen yang jelas yaitu positif dan fitur service (layanan). 
\n Analisis sentimen (sentiment analysis) adalah sebuah teknik atau cara yang digunakan untuk mengidentifikasi bagaimana sebuah sentimen diekspresikan menggunakan teks dan bagaimana sentimen tersebut bisa dikategorikan sebagai sentimen positif maupun sentimen negatif.
\n Demikian pula adanya dengan analisis fitur atau aspek. Dimana analisis fitur ini merupakan subset dari analisis sentimen yang lebih fokus pada proses identifikasi dan analisis sentimen terkait dengan fitur-fitur tertentu dari suatu produk, layanan, atau topik lainnya. 
\n Tujuan utama dalam analisis fitur adalah untuk memahami bagaimana orang merespons atau menilai suatu aspek-aspek spesifik suatu subjek. Misalnya, dalam konteks aplikasi Gojek, aspek-aspek yang akan diteliti mungkin mencakup performa sistem, antarmuka aplikasi, layanan dan lain sebagainya. Analisis fitur memungkinkan untuk melihat pandangan dan sentimen yang lebih rinci terkait dengan berbagai fitur atau aspek yang dianggap penting oleh pengguna. Proses analisis fitur melibatkan identifikasi aspek-aspek yang relevan dalam teks (pengenalan aspek), ekstraksi sentimen terkait aspek tersebut, dan penyajian hasil secara terstruktur. 
\n Dalam proses analisis sentimen berbasis fitur akan dilakukan proses klasifikasi yang akan menggunakan metode algoritma Support Vector Machine. Analisis sentimen ini akan mengklasifikasikan sentimen dari ulasan-ulasan pada aplikasi Gojek, yang kemudian akan dilihat persentasenya terhadap suatu fitur dalam ulasan aplikasi tersebut. 
\n Algoritma Support Vector Machine (SVM) sendiri merupakan metode pembelajaran mesin yang biasa digunakan untuk klasifikasi dan regresi. SVM bekerja dengan membangun bidang pembatas terbaik (hyperplane) yang memaksimalkan jarak antara dua kelas data yang berbeda dalam ruang fitur. SVM bertujuan untuk memisahkan data dengan margin maksimal, yaitu jarak antara hyperplane dan titik-titik data terdekat (support vectors) dari masing-masing kelas.
\n Algoritma SVM akan mengajarkan model menggunakan data pelatihan yang berisi contoh data ulasan yang sudah diklasifikasikan secara manual ke dalam dua kategori sentimen dan beberapa label fitur atau aspek, SVM dapat mempelajari pola-pola dalam teks yang mengindikasikan kelas tertentu. Setelah dilatih, model SVM dapat digunakan untuk mengklasifikasikan ulasan baru ke dalam kategori kelas yang sesuai. 
\n Algoritma ini biasa digunakan untuk menganalisis sentimen karena beberapa alasan yang diantaranya: kemampuan SVM untuk memisahkan dua kelas dengan margin maksimal, penanganan data yang kompleks atau tingkat tinggi, penanganan data yang tidak tersebar secara linier, dan pengendalian overfitting (ketidakmampuan model mengeneralisasi data dengan baik ke data yang belum pernah dilihat sebelumnya). Oleh sebab itu, rancangan ini dibuat untuk melakukan Penerapan Algoritma Support Vector Machine untuk Analisis Sentimen Berbasis FItur pada Ulasan Aplikasi Gojek di Google Play Store.
</div>"""

    image_path = Image.open(r"D:\Kuliah\Semester 9\Skripsi(New)\Data Skripsi\Uji Program\gambar\Gambar_Survei.png")
    image_path2 = Image.open(r"D:\Kuliah\Semester 9\Skripsi(New)\Data Skripsi\Uji Program\gambar\gojek_logo.jpg")
    image_path3 = Image.open(r"D:\Kuliah\Semester 9\Skripsi(New)\Data Skripsi\Uji Program\gambar\google_play_store.png")
    title = "<h1 style='text-align: center;'>Welcome!</h1>"
    subheader = "<h2 style='text-align: center;'> Sistem Website Analisis Sentimen Berbasis Fitur pada Ulasan Aplikasi Gojek di Google Play Store</h2>"
    st.markdown(title, unsafe_allow_html=True)
    st.markdown(subheader, unsafe_allow_html=True)
    st.image(image_path2, caption="Logo Gojek", use_column_width=True)
    st.write(text1, unsafe_allow_html=True)
    st.image(image_path, caption="Hasil Survei Indef", use_column_width=True)
    st.write(text2, unsafe_allow_html=True)
    st.image(image_path3, caption="Logo Google Play Store", use_column_width=True)
    st.write(text3, unsafe_allow_html=True)

if selected == "Analysis":
    #side bar menu
    with st.sidebar:
        # Radio button to choose between uploading CSV or scraping data
        data_source = st.radio("Choose Data Source", ["Upload CSV", "Scrape Data"])
        
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose a file", type = ["csv"])

            if uploaded_file is not None:
                # Check the file type and perform the appropriate action (e.g., read CSV or Excel)
                if uploaded_file.type == 'text/csv':
                    # Read CSV file
                    st.session_state['df'] =  pd.read_csv(uploaded_file)
        
        elif data_source == "Scrape Data":
            with st.form(key = "form1"):
                input = st.text_input(label = "Input URL")
                number = st.number_input(label = "Input Number",min_value=1)
                submit = st.form_submit_button("Enter")
            
            #Proses Scraping dari web
            if submit:
                result, continuation_token = reviews(
                    input,
                    lang='id', # defaultnya adalah 'en'
                    country='id', # defaultnya adalah 'us'
                    sort=Sort.NEWEST, # mengambil ulasan terbaru
                    count = number, # defaultnya adalah 100
                    filter_score_with=None # defaultnya adalah None (artinya semua skor) Gunakan 1, 2, 3, 4, atau 5 untuk memilih skor tertentu
                )

                df_busu = pd.DataFrame(np.array(result),columns=['review'])
                df_busu = df_busu.join(pd.DataFrame(df_busu.pop('review').tolist()))
                st.session_state['df'] = df_busu[['userName','at', 'content', 'score']]

        selected = option_menu(
            menu_title = "Features",  # required
            options = ["Service", "Usability", "Performance"],  # required
            default_index = 0,  # optional
        )
        
    if st.session_state.df is None:
        st.warning("You must upload a file!")
    else:
        df = st.session_state['df']
        load_data(df)
        # Fitur Service dan Non-Service
        # Load the model parameters to get the file paths
        model_folder = r'D:\Kuliah\Semester 9\Skripsi(New)\Data Skripsi\Uji Program\Trained_Model_Service'
        model_parameters = torch.load(r'D:\Kuliah\Semester 9\Skripsi(New)\Data Skripsi\Uji Program\Trained_Model_Service\trained_model_service.pth')
        vectorizer_path = os.path.join(model_folder, 'vectorizer.pkl')
        svm_model_path = os.path.join(model_folder, 'svm_model.pkl')

        # Load the TfidfVectorizer and SVM model
        vectorizer = joblib.load(vectorizer_path)
        svm_model = joblib.load(svm_model_path)

        # Fitur Usability dan Performance
        # Load the model parameters to get the file paths
        model_folder1 = r'D:\Kuliah\Semester 9\Skripsi(New)\Data Skripsi\Uji Program\Trained_Model_Non_Service'
        model_parameters1 = torch.load(r'D:\Kuliah\Semester 9\Skripsi(New)\Data Skripsi\Uji Program\Trained_Model_Non_Service\trained_model_non_service.pth')
        vectorizer_path1 = os.path.join(model_folder1, 'vectorizer1.pkl')
        svm_model_path1 = os.path.join(model_folder1, 'svm_model1.pkl')

        # Load the TfidfVectorizer and SVM model
        vectorizer1 = joblib.load(vectorizer_path1)
        svm_model1 = joblib.load(svm_model_path1)

        # Sentiment
        # Load the model parameters to get the file paths
        model_folder2 = r'D:\Kuliah\Semester 9\Skripsi(New)\Data Skripsi\Uji Program\Trained_Model_Sentiment'
        model_parameters2 = torch.load(r'D:\Kuliah\Semester 9\Skripsi(New)\Data Skripsi\Uji Program\Trained_Model_Sentiment\trained_model_sentiment.pth')
        vectorizer_path2 = os.path.join(model_folder2, 'vectorizer2.pkl')
        svm_model_path2 = os.path.join(model_folder2, 'svm_model2.pkl')

        # Load the TfidfVectorizer and SVM model
        vectorizer2 = joblib.load(vectorizer_path2)
        svm_model2 = joblib.load(svm_model_path2)

        # Replace missing values (NaN) with empty strings
        df['content'] = df['content'].fillna("")

        # Prepare new data for prediction
        data = df['content']  # Replace with your actual new data

        # Use the loaded vectorizer to transform the new data
        data_transformed = vectorizer.transform(data)
        data_transformed2 = vectorizer2.transform(data)

        # Use the loaded SVM model to make predictions on the transformed new data
        predict = svm_model.predict(data_transformed)
        predict2 = svm_model2.predict(data_transformed2)

        # Create a dataframe to store intermediate results
        df_result = pd.DataFrame({'Predicted_Feature': predict, 'Sentiment': predict2, 'Content': data})

        # Create a list to store the final predictions
        final_predictions = []
        predicted_sentiment = []

        # Loop through the intermediate results and make predictions
        for index, row in df_result.iterrows():
            predicted_label = row['Predicted_Feature']
            if predicted_label == 1:
                final_prediction = 'Service'
            elif predicted_label == 2:
                # Use the loaded vectorizer1 to transform the data (wrap in a list to make it iterable)
                data_transformed1 = vectorizer1.transform([row['Content']])
                
                # Use the loaded SVM model1 to make predictions on the transformed data
                predict1 = svm_model1.predict(data_transformed1)
                
                final_prediction = 'Usability' if predict1 == 2 else 'Performance'
            
            predicted_sentiment_label = row['Sentiment']
            if predicted_sentiment_label == 1:
                predicted_sentiments = 'Positive'
            else:
                predicted_sentiments = 'Negative'

            final_predictions.append(final_prediction)
            predicted_sentiment.append(predicted_sentiments)

        # Add the final predictions to the result_df
        df_result['Predicted_Feature'] = final_predictions
        df_result['Sentiment'] = predicted_sentiment

        if selected == "Service":
            st.title(f"User Review for {selected}")

            # Filter the DataFrame to include only rows where 'Predicted_Label' is 'Service'
            service_df = df_result[df_result['Predicted_Feature'] == 'Service']

            # Calculate the percentage based on the count of "1" values in the "label" column
            total_count = len(service_df)
            one_count = (service_df['Sentiment'] == 'Positive').sum()
            percentage = (one_count / total_count) * 100

            # Calculate the remaining percentage
            remaining_percentage = 100 - percentage

            # Create a horizontal bar chart using matplotlib with green and lighter gray bars
            fig, ax = plt.subplots(figsize=(6, 1))
            ax.barh([0], [percentage], color='yellow', height=0.2)  # Green bar for the percentage
            ax.barh([0], [remaining_percentage], left=[percentage], color='lightgray', height=0.2)  # Light gray bar for the remaining percentage

            # Display "Positive" label inside the green bar and above the percentage
            ax.text(percentage / 2, 0, f"Positive\n{percentage:.2f}%", fontsize=8, va='center', color='black', ha='center', rotation=0)

            # Display "Negative" label inside the light gray bar and above the remaining percentage
            ax.text(percentage + (remaining_percentage / 2), 0, f"Negative\n{remaining_percentage:.2f}%", fontsize=8, va='center', color='black', ha='center', rotation=0)

            ax.set_xlim(0, 100)
            ax.axis('off')  # Remove axis

            # Display the chart in Streamlit
            st.pyplot(fig)

            # Display the filtered DataFrame
            st.dataframe(service_df)
            st.subheader("Notes")
            st.write("Total Data = ",total_count)
            st.write("Positive Sentiment = ",one_count)
            st.write("Negative Sentiment = ",total_count - one_count)

        if selected == "Usability":
            st.title(f"User Review for {selected}")
            # Filter the DataFrame to include only rows where 'Predicted_Label' is 'Usability'
            usability_df = df_result[df_result['Predicted_Feature'] == 'Usability']

            # Calculate the percentage based on the count of "1" values in the "label" column
            total_count = len(usability_df)
            one_count = (usability_df['Sentiment'] == 'Positive').sum()
            percentage = (one_count / total_count) * 100

            # Calculate the remaining percentage
            remaining_percentage = 100 - percentage

            # Create a horizontal bar chart using matplotlib with green and lighter gray bars
            fig, ax = plt.subplots(figsize=(6, 1))
            ax.barh([0], [percentage], color='yellow', height=0.2)  # Green bar for the percentage
            ax.barh([0], [remaining_percentage], left=[percentage], color='lightgray', height=0.2)  # Light gray bar for the remaining percentage

            # Display "Positive" label inside the green bar and above the percentage
            ax.text(percentage / 2, 0, f"Positive\n{percentage:.2f}%", fontsize=8, va='center', color='black', ha='center', rotation=0)

            # Display "Negative" label inside the light gray bar and above the remaining percentage
            ax.text(percentage + (remaining_percentage / 2), 0, f"Negative\n{remaining_percentage:.2f}%", fontsize=8, va='center', color='black', ha='center', rotation=0)

            ax.set_xlim(0, 100)
            ax.axis('off')  # Remove axis

            # Display the chart in Streamlit
            st.pyplot(fig)
            
            # Display the filtered DataFrames
            st.dataframe(usability_df)
            st.subheader("Notes")
            st.write("Total Data = ",total_count)
            st.write("Positive Sentiment = ",one_count)
            st.write("Negative Sentiment = ",total_count - one_count)

        if selected == "Performance":
            st.title(f"User Review for {selected}")
            # Filter the DataFrame to include only rows where 'Predicted_Feature' is 'Performance'
            performance_df = df_result[df_result['Predicted_Feature'] == 'Performance']

            # Calculate the percentage based on the count of "1" values in the "label" column
            total_count = len(performance_df)
            one_count = (performance_df['Sentiment'] == 'Positive').sum()
            percentage = (one_count / total_count) * 100

            # Calculate the remaining percentage
            remaining_percentage = 100 - percentage

            # Create a horizontal bar chart using matplotlib with green and lighter gray bars
            fig, ax = plt.subplots(figsize=(6, 1))
            ax.barh([0], [percentage], color='yellow', height=0.2)  # Green bar for the percentage
            ax.barh([0], [remaining_percentage], left=[percentage], color='lightgray', height=0.2)  # Light gray bar for the remaining percentage

            # Display "Positive" label inside the green bar and above the percentage
            ax.text(percentage / 2, 0, f"Positive\n{percentage:.2f}%", fontsize=8, va='center', color='black', ha='center', rotation=0)

            # Display "Negative" label inside the light gray bar and above the remaining percentage
            ax.text(percentage + (remaining_percentage / 2), 0, f"Negative\n{remaining_percentage:.2f}%", fontsize=8, va='center', color='black', ha='center', rotation=0)

            ax.set_xlim(0, 100)
            ax.axis('off')  # Remove axis

            # Display the chart in Streamlit
            st.pyplot(fig)

            # Display the filtered DataFrames
            st.dataframe(performance_df)
            st.subheader("Notes")
            st.write("Total Data = ",total_count)
            st.write("Positive Sentiment = ",one_count)
            st.write("Negative Sentiment = ",total_count - one_count)

if selected == "About Me":
    st.title(f"{selected}")
    text = """<div style='text-align: justify;'>Halo, saya <b>Jonathan Adrian Wibowo!<b>
    \n Saya merupakan mahasiswa tingkat akhir di Universitas Tarumanagara dari Fakultas Teknologi Informasi, Jurusan Teknik Informatika. Saya sedang membuat rancangan sistem berbasis website
    untuk Analisis Sentimen Berbasis Fitur pada Ulasan Aplikasi Gojek di Google Play Store. Pembuatan rancangan ini guna menyelesaikan tugas akhir saya di bidang Data Engineering. Besar harapan saya
    hasil rancangan ini dapat bermanfaat bagi banyak pihak.
    </div>"""
    st.write(text, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    # Tambahkan elemen UI ke sidebar kolom pertama
    with col1:
        st.subheader("Pembimbing")
        st.write("Viny Christanti Mawardi, S.Kom., M.Kom.")
        st.write("Tri Sutrisno, S.Si., M.Sc.")
        st.subheader("Kontak")
        st.write("Email: jonathan.535190065@stu.untar.ac.id")
        st.write("Linkedin: linkedin.com/in/jonathanadrianwibowo/")

    # Tambahkan elemen UI ke sidebar kolom kedua
    with col2:
        image_path1 = Image.open(r"D:\Kuliah\Semester 9\Skripsi(New)\Data Skripsi\Uji Program\gambar\logo_fti.png")
        st.image(image_path1, use_column_width=True)