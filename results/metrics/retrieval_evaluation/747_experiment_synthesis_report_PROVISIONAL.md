# Ringkasan Hasil Mini-Project Multimodal Retrieval (PROVISIONAL)

## Metode

Dataset terdiri atas 62 gallery, 21 query validation, dan 21 query test.

Fitur visual menggunakan HSV, GLCM, dan Hu Moments. Fitur teks menggunakan TF-IDF. Similarity visual berbasis jarak Euclidean yang dikalibrasi menggunakan gallery, sedangkan similarity teks menggunakan cosine similarity.

Relevansi ditentukan berdasarkan kesamaan kelas tanaman.

Pemilihan konfigurasi dilakukan hanya menggunakan validation dengan metrik utama mAP@5.

Konfigurasi terpilih adalah **Unigram + Bigram**, **alpha=0.3**, menggunakan metode **late fusion visual dan teks**.

Konfigurasi ini tidak diubah setelah melihat hasil test.

## Hasil Validation dan Test

| Metrik (K=5) | Validation | Test (Provisional) | Test - Validation |
|---|---:|---:|---:|
| Precision@5 | 0.7905 | 0.8381 | +0.0476 |\n| Recall@5 | 0.3148 | 0.3341 | +0.0193 |\n| MRR@5 | 0.8532 | 1.0000 | +0.1468 |\n| mAP@5 | 0.7200 | 0.8087 | +0.0887 |\n| nDCG@5 | 0.7854 | 0.8721 | +0.0867 |\n

Pada test, Top-1 benar untuk **21/21** query (100.0%).

Sementara itu, Top-5 memuat sedikitnya satu kandidat relevan untuk **21/21** query (100.0%).

Kedua indikator tersebut menunjukkan keberhasilan retrieval berdasarkan kesamaan kelas, bukan akurasi suatu model klasifikasi terlatih.

## Interpretasi

Selisih antara validation dan test bersifat deskriptif, bukan uji signifikansi ataupun bukti kausal bahwa model mengalami penurunan atau peningkatan generalisasi.

Validation digunakan untuk memilih bobot, sehingga skor validation berpotensi optimistis.

## Keterbatasan

- Kandidat pasangan gambar mirip antar-split: **44**. Kemiripan visual tidak otomatis membuktikan gambar berasal dari spesimen yang sama.
- Kelompok deskripsi identik antar-split: **4**. Potensi kemudahan retrieval akibat pengulangan teks perlu dipertimbangkan.
- Provenansi spesimen atau pengambilan gambar belum terverifikasi secara independen.
- Kesetaraan ekstraktor fitur visual dengan Notebook 03 belum terverifikasi secara independen.
- Hasil test tidak boleh digunakan untuk memilih ulang bobot alpha atau konfigurasi TF-IDF.

## Kesimpulan Sementara

Konfigurasi yang dipilih berdasarkan validation telah diterapkan secara konsisten pada test dan menghasilkan laporan metrik serta analisis kesalahan retrieval.

Hasil tersebut merupakan analisis eksploratif atau provisional dan belum dapat diklaim sebagai estimasi kinerja independen yang bebas kebocoran data.

Penyelesaian audit diperlukan sebelum menyusun kesimpulan final.
