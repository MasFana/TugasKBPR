# Laporan Praktikum

**Nama:** Andhika Rachmat  
**NIM:** 222410103079  
**Kelas:** A  
**Praktikum:** 1

## 1. Instalasi Package Python

### a. Jika pip belum terinstal

1. Unduh `get-pip.py` dari [tautan ini](https://bootstrap.pypa.io/getpip.py).
2. Jalankan perintah berikut di terminal:
   ```bash
   python get-pip.py
   ```

### b. Instalasi Package dengan pip

Jika pip sudah terinstal, Anda bisa menginstal package menggunakan:
```bash
pip install <nama_package>
```
Contoh untuk menginstal `numpy`:
```bash
pip install numpy
```

## 2. Pemanggilan Module di dalam Package Python

Contoh pemanggilan module:
```python
import numpy.core
result = numpy.core.subtract(3, 2)
print(result)
```

Atau menggunakan alias:
```python
from numpy.core import subtract as sub
result = sub(3, 2)
print(result)
```

## 3. Variabel Logika, Constraint, dan Goal

Contoh penggunaan variabel logika dan ekspresi logika:
```python
a = True
b = False

c = a and b  # AND
d = a or b   # OR
e = not a    # NOT

print(f"a AND b: {c}")
print(f"a OR b: {d}")
print(f"NOT a: {e}")
```

## 4. Fakta dan Relasi

Contoh deklarasi fakta dan relasi:
```python
# Contoh deklarasi fakta menggunakan format Prolog-like
fact = {
    "Ayah": ("Musa", "Azlan"),
    "Berbatasan": ("Jember", "Banyuwangi")
}
```

## 5. Pustaka Dasar

### a. TensorFlow

Instal TensorFlow:
```bash
pip install tensorflow
```

Contoh kode:
```python
import tensorflow as tf

tensor_a = tf.constant([[1, 2, 3], [4, 5, 6]])
tensor_b = tf.constant([[7, 8, 9], [10, 11, 12]])

tensor_sum = tf.add(tensor_a, tensor_b)
tensor_diff = tf.subtract(tensor_a, tensor_b)
tensor_product = tf.multiply(tensor_a, tensor_b)

print("Tensor A + Tensor B:\n", tensor_sum.numpy())
print("Tensor A - Tensor B:\n", tensor_diff.numpy())
print("Tensor A * Tensor B:\n", tensor_product.numpy())
```

**Penjelasan TensorFlow:**  
TensorFlow adalah pustaka open-source untuk komputasi numerik dan pembelajaran mesin. Ia menyediakan API untuk membangun dan melatih model machine learning, dengan dukungan untuk tensor, grafik aliran data, dan optimisasi.

### b. NumPy

Instal NumPy:
```bash
pip install numpy
```

Contoh kode:
```python
import numpy as np

array_a = np.array([1, 2, 3, 4, 5])
array_b = np.array([6, 7, 8, 9, 10])

array_sum = array_a + array_b
array_product = array_a * array_b
array_mean = np.mean(array_a)

print("Array A:", array_a)
print("Array B:", array_b)
print("Penjumlahan Array A dan Array B:", array_sum)
print("Perkalian Array A dan Array B:", array_product)
print("Rata-rata Array A:", array_mean)
```

**Penjelasan NumPy:**  
NumPy adalah pustaka untuk operasi matematis dan komputasi numerik yang menyediakan struktur data array multidimensi dan berbagai fungsi matematis untuk operasi di atas array.

### c. Pandas

Instal Pandas:
```bash
pip install pandas
```

Contoh kode:
```python
import pandas as pd

data = {
    'Nama': ['Abi', 'Budi', 'Caca'],
    'Usia': [25, 30, 35],
    'Kota': ['Jember', 'Banyuwangi', 'Bali']
}

df = pd.DataFrame(data)

print("DataFrame:")
print(df)

print("\nStatistik Deskriptif:")
print(df.describe())

print("\nNama dan Usia:")
print(df[['Nama', 'Usia']])
```

**Penjelasan Pandas:**  
Pandas adalah pustaka untuk manipulasi dan analisis data, terutama dengan struktur data DataFrame yang memungkinkan pemrosesan data tabel dengan cara yang efisien.

### d. Matplotlib

Instal Matplotlib:
```bash
pip install matplotlib
```

Contoh kode:
```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [2, 4, 6]

plt.plot(x, y, label='Garis')
plt.xlabel('Sumbu X')
plt.ylabel('Sumbu Y')
plt.title('Contoh Grafik Garis')
plt.legend()
plt.show()
```

**Penjelasan Matplotlib:**  
Matplotlib adalah pustaka untuk visualisasi data yang memungkinkan pembuatan grafik dan plot untuk analisis data dan hasil model.

## Tugas Mandiri

### 1. Matriks A dan B

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matriks A:")
print(A)

print("Matriks B:")
print(B)

print("Penjumlahan A dan B:")
print(A + B)

print("Pengurangan A dan B:")
print(A - B)

print("Perkalian A dan B:")
print(A @ B)

print("Transpose Matriks A:")
print(A.T)

print("Transpose Matriks B:")
print(B.T)
```

### 2. Visualisasi Grafik Garis

```python
import matplotlib.pyplot as plt

# Data
semesters = [1, 2, 3, 4, 5, 6, 7, 8]
ipk = [3.0, 3.2, 3.5, 3.7, 3.8, 3.9, 4.0, 4.0]
sks = [20, 22, 24, 26, 28, 30, 32, 34]

plt.plot(semesters, ipk, label='IPK', marker='o')
plt.plot(semesters, sks, label='Jumlah SKS', marker='o')

plt.xlabel('Semester')
plt.ylabel('Nilai')
plt.title('IPK dan Jumlah SKS Mahasiswa Fasilkom')
plt.legend()
plt.grid(True)
plt.show()
```

## Pengumpulan

Format penamaan file: `Kelas_Praktikum1_222410103079_Andhika_Rachmat.md`  
Deadline Pengumpulan: Senin, 23 September 2024 pukul 23.59  
Link Pengumpulan: [Google Forms](https://forms.gle/eJscxriJdkbuSv3P6)
