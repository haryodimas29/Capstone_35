# Capstone_35
Pada Github ini terdapat dataset untuk capstone project kami, Kode data science & AI, serta deploy model ke Web App Streamlit
## Dataset
https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/data
Dataset ini terdiri dari 10.000 titik data yang disimpan sebagai baris dengan 14 fitur dalam kolom:
UID: Identifier unik yang berada dalam rentang 1 hingga 10.000.
productID: Terdiri dari huruf L, M, atau H untuk variasi kualitas produk rendah (50% dari semua produk), sedang (30%), dan tinggi (20%), serta nomor serial yang spesifik untuk variasi.
Air Temperature [K]: Suhu udara yang dihasilkan menggunakan proses berjalan acak kemudian dinormalisasi menjadi standar deviasi 2 K sekitar 300 K.
Process Temperature [K]: Suhu proses yang dihasilkan menggunakan proses berjalan acak dinormalisasi menjadi standar deviasi 1 K, ditambahkan dengan suhu udara plus 10 K.
Rotational Speed [rpm]: Kecepatan rotasi yang dihitung dari daya 2860 W, ditutup dengan noise yang tersebar normal.
Torque [Nm]: Nilai torsi yang tersebar normal sekitar 40 Nm dengan σ = 10 Nm dan tidak ada nilai negatif.
Tool Wear [min]: Penggunaan alat yang dihitung dengan menambahkan 5/3/2 menit penggunaan alat pada alat yang digunakan dalam proses berdasarkan variasi kualitas produk.
Machine Failure: Label yang menunjukkan apakah mesin telah gagal dalam titik data ini untuk salah satu dari berikutnya:
Target: Gagal atau Tidak
Failure Type: Tipe Kegagalan

## Data Science dan AI
Link Google Colab: https://colab.research.google.com/drive/1N41FPiK2ul2sigUepgZCMoow88xK5rKp?usp=sharing
Link IBM Watson: https://jp-tok.dataplatform.cloud.ibm.com/analytics/notebooks/v2/cdcb9304-411f-4d7c-86db-2bccacabfaa5/view?access_token=a47de5637e676c5bce241e70a26aeb1643cd2c37d276bd4cf1b930d94aacde8a&context=cpdaas 
### 1. Pengumpulan Data
- Mengambil dataset CSV dengan Pandas
- Melakukan Exploratory Data Analysis (EDA) dengan statistika dan visualisasi data yang menggunakan library pandas, matplotlib, dan seaborn 
### 2. Pra-Pemrosesan Data
- Melakukan pembersihan data dengan mengecek nilai hilang, duplikat, menghapuskan outlier, menghapuskan kolom yang tidak diperlukan
- Transformasi data dengan mengkodekan data kategoris, dan membagi dataset menjadi Training (70%), Testing(20%), dan validation(10%).
### 3. Training Data
- Training Data dengan model-model machine learning dan deep learning ini ini:
####  Random Forest (RF):
adalah algoritma klasifikasi yang menggunakan teknik ensemble learning. Algoritma ini bekerja dengan cara menggabungkan beberapa model klasifikasi yang dibuat dengan menggunakan teknik sampling random dan fitur yang dipilih secara acak. Dengan cara ini, RF dapat meningkatkan akurasi prediksi dan robustness terhadap variasi data.
  
#### Gradient Boosting (GB): 
adalah algoritma klasifikasi yang menggunakan teknik ensemble learning dan gradient descent untuk memperbaiki prediksi model. Algoritma ini bekerja dengan cara membuat model klasifikasi pertama menggunakan data yang tersedia, lalu menggunakan gradient descent untuk memperbaiki prediksi model pertama. Hasilnya adalah model yang lebih akurat dan robust terhadap noise dan variasi data.

#### Support Vector Machine (SVM):
adalah algoritma klasifikasi yang menggunakan teknik kernel untuk memperbaiki prediksi model. Algoritma ini bekerja dengan cara menggunakan kernel untuk memperbaiki prediksi model, lalu membagi data menjadi dua kelas dengan menggunakan hyperplane yang dipilih secara optimal. Hasilnya adalah model yang lebih akurat dan robust terhadap noise dan variasi data.

#### Multilayer Perceptron (MLP):
adalah algoritma klasifikasi yang menggunakan teknik neural network untuk memperbaiki prediksi model. Algoritma ini bekerja dengan cara membuat beberapa layer yang berisi neuron untuk memperbaiki prediksi, lalu menggunakan neuron untuk memperbaiki prediksi dalam setiap layer. Hasilnya adalah model yang lebih akurat dan robust terhadap noise dan variasi data.

- Akurasi masing-masing model akan dicek. Setelah itu, Model dengan akurasi terbaik dilakukan Hyperparameter Tuning untuk meningkatkan performa model
### 4. Evaluasi Data
- Model dievaluasi dengan menggunakan:
#### Classification Report
Classification report adalah laporan yang menampilkan informasi tentang performansi model klasifikasi. Laporan ini biasanya berisi beberapa indikator, seperti:

Accuracy: Akurasi model, yang didefinisikan sebagai jumlah prediksi yang benar dibandingkan dengan jumlah total prediksi.

Precision: Presisi model, yang didefinisikan sebagai jumlah prediksi yang benar dibandingkan dengan jumlah total prediksi yang dianggap benar.

Recall: Recall model, yang didefinisikan sebagai jumlah prediksi yang benar dibandingkan dengan jumlah total prediksi yang seharusnya benar.

F1-score: F1-score model, yang didefinisikan sebagai rata-rata dari precision dan recall. Laporan ini dapat membantu dalam mengevaluasi performansi model dan membandingkan dengan model lainnya.
#### Confusion Matrix
Confusion matrix adalah tabel yang menampilkan hasil prediksi model klasifikasi. Tabel ini berisi empat elemen:

True Positive (TP): Jumlah prediksi yang benar untuk kelas yang dianggap benar.

True Negative (TN): Jumlah prediksi yang benar untuk kelas yang dianggap salah.

False Positive (FP): Jumlah prediksi yang salah untuk kelas yang dianggap salah.

False Negative (FN): Jumlah prediksi yang salah untuk kelas yang dianggap benar.

Confusion matrix dapat membantu dalam mengevaluasi performansi model dan memahami kesalahan yang terjadi.
#### Stratified Kfold
Stratified Kfold adalah teknik cross-validation yang digunakan untuk membagi data menjadi beberapa bagian yang seimbang. Teknik ini digunakan untuk memastikan bahwa setiap bagian memiliki proporsi kelas yang seimbang, sehingga performansi model dapat diperkirakan dengan lebih akurat.
Stratified Kfold bekerja dengan cara membagi data menjadi beberapa bagian yang seimbang, kemudian menggunakan setiap bagian untuk melatih dan menguji model. Hasilnya adalah performansi model yang lebih akurat dan lebih stabil.
Dalam beberapa penelitian, stratified Kfold digunakan untuk membandingkan performansi model klasifikasi dan memahami kesalahan yang terjadi.
### 5. Deploy ke Streamlit
- Model yang telah dilakukan training dan testing akan di deploy ke streamlit
- Download file .pkl dari model

## App Streamlit
