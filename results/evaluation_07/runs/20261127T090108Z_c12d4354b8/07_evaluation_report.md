# Laporan Evaluasi Akhir — Notebook 07

**Proyek:** Multimodal Medicinal Leaf Retrieval
**Sumber:** Notebook 06 run `20261127T090108Z_c12d4354b8` (penemuan run: LATEST_COMPLETED_RUN.json)
**Dibuat:** 2026-11-27T10:24:53+00:00 · Notebook 07 `07_eval_final_v1.0`
**Status test:** **PROVISIONAL_AUDIT_UNRESOLVED** — bukan estimasi independen.

## 1. Ringkasan
- Data: 104 sampel, 5 kelas; gallery 62, validation 21, test 21.
- Konfigurasi terkunci (validation-only): TF-IDF Unigram + Bigram, α = 0,3 (visual), 0,7 (teks).
- Verifikasi independen: 2604 baris metrik per query dan
  284 baris macro direproduksi dari ranking tersimpan (selisih maks
  0.0e+00).

## 2. Perbandingan metode @5
### Validation (dipakai untuk seleksi)
| Metode | Precision@5 | Recall@5 | MRR@5 | mAP@5 | nDCG@5 |
|---|---|---|---|---|---|
| Visual Only | 0.6000 | 0.2374 | 0.9048 | 0.5522 | 0.6525 |
| Text Only - Unigram | 0.6952 | 0.2748 | 0.7738 | 0.6246 | 0.6988 |
| Text Only - Unigram + Bigram | 0.7619 | 0.3037 | 0.8492 | 0.6911 | 0.7623 |
| Weighted Fusion - Unigram + Bigram (alpha=0.5) | 0.7429 | 0.2942 | 0.8770 | 0.6933 | 0.7565 |
| Weighted Fusion - Unigram + Bigram (alpha=0.3) | 0.7714 | 0.3061 | 0.8532 | 0.7052 | 0.7715 |

### Test — PROVISIONAL_AUDIT_UNRESOLVED (hanya konfigurasi yang diranking Notebook 06)
| Metode | Precision@5 | Recall@5 | MRR@5 | mAP@5 | nDCG@5 |
|---|---|---|---|---|---|
| Visual Only | 0.6286 | 0.2538 | 0.9444 | 0.5878 | 0.6940 |
| Text Only - Unigram + Bigram | 0.8381 | 0.3333 | 0.9524 | 0.7908 | 0.8509 |
| Weighted Fusion - Unigram + Bigram (alpha=0.3) | 0.8286 | 0.3294 | 1.0000 | 0.7941 | 0.8573 |

### Selisih mAP@5 fusion terhadap baseline
| partition | baseline | fusion | baseline_value | delta_pp | delta_rel_percent |
|---|---|---|---|---|---|
| validation | Visual Only | 0.7052 | 0.5522 | 15.3016 | 27.7091 |
| validation | Text Only - Unigram | 0.7052 | 0.6246 | 8.0635 | 12.9098 |
| validation | Text Only - Unigram + Bigram | 0.7052 | 0.6911 | 1.4127 | 2.0441 |
| validation | Weighted Fusion - Unigram + Bigram (alpha=0.5) | 0.7052 | 0.6933 | 1.1905 | 1.7170 |
| test | Visual Only | 0.7941 | 0.5878 | 20.6349 | 35.1067 |
| test | Text Only - Unigram + Bigram | 0.7941 | 0.7908 | 0.3333 | 0.4215 |

**Interpretasi.** Pada validation, fusion terpilih memiliki mAP@5 0,7052, yaitu
+1,41 pp (+2,0% relatif) terhadap Text Only - Unigram + Bigram dan
+15,30 pp (+27,7% relatif) terhadap Visual Only. Fusion lebih tinggi
pada 5 dari 5 metrik @5 dibanding Text Only - Unigram + Bigram.
Keunggulan validation ini sebagian berasal dari proses seleksi itu sendiri (α dipilih pada data yang sama).
Pada test (PROVISIONAL_AUDIT_UNRESOLVED), fusion lebih tinggi pada 3 dari 5 metrik dibanding Text Only - Unigram + Bigram
(lebih rendah pada Precision@5, Recall@5). Text Only - Unigram dan Fusion α=0,5 tidak diranking pada
test oleh Notebook 06, sehingga tidak dibandingkan di test.

## 3. Performa konfigurasi terkunci per K
| k | mAP validation | mAP test | nDCG validation | nDCG test | Precision validation | Precision test | Recall validation | Recall test | MRR validation | MRR test |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7619 | 1.0000 | 0.7619 | 1.0000 | 0.7619 | 1.0000 | 0.0603 | 0.0808 | 0.7619 | 1.0000 |
| 3 | 0.7460 | 0.8307 | 0.7925 | 0.8736 | 0.8095 | 0.8413 | 0.1935 | 0.2006 | 0.8413 | 1.0000 |
| 5 | 0.7052 | 0.7941 | 0.7715 | 0.8573 | 0.7714 | 0.8286 | 0.3061 | 0.3294 | 0.8532 | 1.0000 |
| 10 | 0.5750 | 0.6358 | 0.6899 | 0.7490 | 0.6524 | 0.6857 | 0.5149 | 0.5386 | 0.8532 | 1.0000 |

**Interpretasi.** Pada konfigurasi terkunci (Unigram + Bigram, α = 0,3),
mAP@5 validation = **0,7052** dan nDCG@5 = **0,7715**.
Precision turun dari 0,7619 (K=1) menjadi 0,6524 (K=10), sedangkan
Recall naik menjadi 0,5149 pada K=10. Pola ini wajar: setiap kelas memiliki
11–15 item relevan di gallery, sehingga Recall@K dibatasi oleh K/R.
Pada test (**PROVISIONAL_AUDIT_UNRESOLVED**), mAP@5 = 0,7941; 5 dari 5 metrik @5 lebih tinggi
daripada validation. Angka test hanya bersifat deskriptif untuk konfigurasi yang sudah dikunci dan bukan
estimasi generalisasi independen.

## 4. Kontribusi fusion (paired per query, K=5, terhadap Text Only - Unigram + Bigram)
| partition | metric | mean_difference | ci95_low | ci95_high | wins | ties | losses | sign_test_p |
|---|---|---|---|---|---|---|---|---|
| validation | precision@5 | 0.0095 | -0.0190 | 0.0381 | 2 | 18 | 1 | 1.0000 |
| validation | recall@5 | 0.0024 | -0.0087 | 0.0127 | 2 | 18 | 1 | 1.0000 |
| validation | mrr@5 | 0.0040 | -0.0675 | 0.0714 | 2 | 18 | 1 | 1.0000 |
| validation | ap@5 | 0.0141 | -0.0195 | 0.0532 | 4 | 15 | 2 | 0.6875 |
| validation | ndcg@5 | 0.0092 | -0.0226 | 0.0467 | 4 | 15 | 2 | 0.6875 |
| test | precision@5 | -0.0095 | -0.0286 | 0.0000 | 0 | 20 | 1 | 1.0000 |
| test | recall@5 | -0.0040 | -0.0119 | 0.0000 | 0 | 20 | 1 | 1.0000 |
| test | mrr@5 | 0.0476 | 0.0000 | 0.1190 | 2 | 19 | 0 | 0.5000 |
| test | ap@5 | 0.0033 | -0.0181 | 0.0238 | 2 | 18 | 1 | 1.0000 |
| test | ndcg@5 | 0.0064 | -0.0128 | 0.0305 | 2 | 18 | 1 | 1.0000 |

**Interpretasi.** Kontribusi visual dinilai sebagai selisih fusion terhadap Text Only - Unigram + Bigram (konfigurasi teks yang sama, α = 0 → α = 0,3).
- **Validation**: ΔmAP@5 = +1,41 pp (CI 95% bootstrap −1,95 pp s.d. +5,32 pp; mencakup nol), ΔnDCG@5 = +0,92 pp, ΔMRR@5 = +0,40 pp, ΔPrecision@5 = +0,95 pp. Fusion lebih tinggi pada 4 query, sama pada 15, lebih rendah pada 2 (AP@5; uji tanda p = 0,688).
- **Test**: ΔmAP@5 = +0,33 pp (CI 95% bootstrap −1,81 pp s.d. +2,38 pp; mencakup nol), ΔnDCG@5 = +0,64 pp, ΔMRR@5 = +4,76 pp, ΔPrecision@5 = −0,95 pp. Fusion lebih tinggi pada 2 query, sama pada 18, lebih rendah pada 1 (AP@5; uji tanda p = 1,000).

Pada validation keuntungan muncul di semua metrik @5. Interval kepercayaan bootstrap dan uji tanda bersifat deskriptif: validation telah dipakai untuk memilih α (bias seleksi), sedangkan test berstatus provisional dengan 21 query, sehingga tidak ada klaim signifikansi statistik.

## 5. Sensitivitas α (validation)
| text_configuration | best_alpha_by_mAP@5 | best_mAP@5 | mAP@5_text_only(alpha=0) | mAP@5_visual_only(alpha=1) | mAP@5_range | alphas_within_0.01_of_best | fusion_alphas_beating_text_only |
|---|---|---|---|---|---|---|---|
| Unigram | 0.7000 | 0.6727 | 0.6246 | 0.5522 | 0.1205 | 0,6; 0,7; 0,8 | 7 |
| Unigram + Bigram | 0.3000 | 0.7052 | 0.6911 | 0.5522 | 0.1530 | 0,3; 0,4 | 3 |

**Interpretasi.** Untuk Unigram + Bigram, mAP@5 validation bergerak dari 0,6911
(α=0, teks saja) ke 0,5522 (α=1, visual saja), dengan puncak
0,7052 pada α = 0,3. Sebanyak
3 dari 9 nilai α campuran melampaui teks saja. Nilai α dalam 0,01 dari puncak:
0,3; 0,4. Konfigurasi Unigram berpuncak di
0,6727 (α = 0,7).

Pola ini **konsisten dengan** teks sebagai modalitas utama (skor teks saja lebih tinggi daripada visual saja) dan visual sebagai sinyal pelengkap pada bobot kecil–sedang. Bobot visual yang makin besar menggeser kinerja menuju level visual saja.
Ini adalah deskripsi kurva validation, bukan bukti mekanisme; dengan 21 query validation, perbedaan antar-α yang kecil
dapat berubah pada data lain. Notebook 07 tidak memilih α baru.

## 6. Validation vs test
**Interpretasi.** Untuk konfigurasi terkunci, 20 dari 20 kombinasi metrik×K bernilai lebih tinggi pada test;
Δ mAP@5 = +8,89 pp. Selisih ini **tidak** dapat dibaca sebagai bukti generalisasi atau signifikansi:
(1) validation dipakai untuk memilih α sehingga tidak independen dari konfigurasi; (2) test sudah pernah diperiksa
(status PROVISIONAL_AUDIT_UNRESOLVED); (3) masing-masing partisi hanya 21 dan 21 query, sehingga
satu query mengubah rata-rata sekitar 4,8 pp untuk metrik bernilai 0/1; (4) temuan near-duplicate
dan deskripsi identik lintas split (Cell 12) dapat menaikkan skor test.

## 7. Evaluasi per kelas
| partition | class | queries | gallery_relevant_R | Precision@5 | Recall@5 | mAP@5 | nDCG@5 | Top-1 correct rate | Top-5 hit rate |
|---|---|---|---|---|---|---|---|---|---|
| validation | Mangkok | 4 | 12 | 1.0000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | Salam | 4 | 11 | 0.4000 | 0.1818 | 0.2017 | 0.3193 | 0.0000 | 1.0000 |
| validation | Sambang Darah | 4 | 12 | 0.6500 | 0.2708 | 0.6042 | 0.7216 | 1.0000 | 1.0000 |
| validation | Sirih | 4 | 12 | 0.8500 | 0.3542 | 0.8500 | 0.8979 | 1.0000 | 1.0000 |
| validation | Tapak Kuda | 5 | 15 | 0.9200 | 0.3067 | 0.8373 | 0.8894 | 0.8000 | 1.0000 |
| test | Mangkok | 4 | 12 | 1.0000 | 0.4167 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | Salam | 4 | 11 | 0.5500 | 0.2500 | 0.4358 | 0.6065 | 1.0000 | 1.0000 |
| test | Sambang Darah | 4 | 12 | 0.7000 | 0.2917 | 0.6625 | 0.7633 | 1.0000 | 1.0000 |
| test | Sirih | 4 | 12 | 0.8500 | 0.3542 | 0.8208 | 0.8809 | 1.0000 | 1.0000 |
| test | Tapak Kuda | 5 | 15 | 1.0000 | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Interpretasi.** Kelas terlemah berdasarkan mAP@5 adalah **Salam** pada validation
(mAP@5 = 0,2017, Top-1 = 0,0000) dan **Salam** pada test
(mAP@5 = 0,4358). Jumlah query per kelas hanya 4–5, sehingga satu query dapat menggeser
rata-rata kelas sebesar 20–25 pp (untuk metrik bernilai 0/1). Jumlah item relevan di gallery (R) juga berbeda antarkelas
(11–15); karena itu Recall@5 tidak sebanding langsung antarkelas
(batas atasnya 5/R). Perbedaan antarkelas sebaiknya dibaca sebagai indikasi, bukan peringkat kelas yang stabil.

## 8. Kesalahan retrieval
**Interpretasi.** Matriks ini menggambarkan kelas tetangga top-1 hasil retrieval, bukan prediksi classifier.
Top-1 benar: validation 16/21,
test 21/21 (PROVISIONAL_AUDIT_UNRESOLVED). Pasangan yang paling sering tertukar: Salam → Sambang Darah (4 query, validation).
Dari 5 kesalahan top-1 validation, ranking Visual Only (tabel 646) benar di top-1 pada 3 query dan Text Only benar pada 1 query; 2 query memiliki fitur visual di luar rentang gallery (tabel 637). Ini adalah keterkaitan deskriptif, bukan bukti penyebab. Query dengan setidaknya satu item relevan di Top-5:
21/21 (validation)
dan 21/21 (test).

## 9. Kebocoran data dan robustness
| finding | count | category |
|---|---|---|
| Kandidat near-duplicate (dHash) total | 80 | Perlu ditinjau |
| Kandidat near-duplicate lintas split | 44 | Perlu ditinjau |
| Query val/test dengan pasangan near-duplicate di gallery | 12 | Terkonfirmasi (fakta terukur) |
| Kelompok deskripsi identik lintas split | 4 | Terkonfirmasi (fakta terukur) |
| Query test dengan top-1 berdeskripsi identik | 3 | Terkonfirmasi (fakta terukur) |
| Query test dengan top-1 kandidat near-duplicate | 2 | Perlu ditinjau |
| Identitas spesimen/sesi foto tidak dideklarasikan | 1 | Tidak dapat diselesaikan dari CSV |
| Test pernah diperiksa pada run sebelumnya | 1 | Terkonfirmasi (fakta terukur) |
| Query dengan fitur visual di luar rentang gallery (|z|>10) | 7 | Terkonfirmasi (fakta terukur) |
| Audit piksel gambar pada run sumber | 0 | Terkonfirmasi (fakta terukur) |

**Interpretasi.**

*Terkonfirmasi (fakta terukur):* 4 kelompok deskripsi identik melintasi split;
3 query test mendapat tetangga top-1 dengan deskripsi identik;
test sudah pernah diperiksa; 7 query memiliki fitur visual di luar rentang gallery.

*Perlu ditinjau:* 44 kandidat near-duplicate lintas split (status: {'AWAITING_PAIR_REVIEW': 44}).
Query bertanda risiko (deskripsi identik atau pasangan near-duplicate di gallery) memiliki rata-rata AP@5
0,5567 vs 0,7647 tanpa tanda (validation) dan
0,7493 vs 0,8278 (test). Pada data ini, query bertanda tidak memperoleh skor rata-rata lebih tinggi daripada query tanpa tanda, sehingga tidak tampak inflasi skor yang jelas dari tanda tersebut.
Perbandingan ini deskriptif dengan jumlah query kecil; ia tidak membuktikan ada atau tidaknya kebocoran.

*Tidak dapat diselesaikan dari CSV:* identitas spesimen, apakah pasangan near-duplicate berasal dari daun yang sama, dan
independensi test. Hal ini membutuhkan pemeriksaan gambar/metadata pengambilan foto oleh reviewer.

## Pembahasan hasil

Seluruh angka berasal dari run Notebook 06 `20261127T090108Z_c12d4354b8` dan telah diverifikasi ulang dari ranking tersimpan (Cell 3).
Status test: **PROVISIONAL_AUDIT_UNRESOLVED**.

**1. Efektivitas modalitas visual.** Visual Only mencapai mAP@5 0,5522 pada validation dan 0,5878 pada test,
terendah di antara metode yang dibandingkan (Gambar F02, Tabel T04). Fitur HSV/GLCM/Hu bersifat global; beberapa query memiliki
nilai histogram warna di luar rentang gallery (Cell 12), yang dapat membuat skor visualnya hampir seragam terhadap seluruh gallery.

**2. Efektivitas modalitas teks.** Text Only lebih tinggi daripada Visual Only: Unigram 0,6246 dan Unigram + Bigram
0,6911 (mAP@5 validation). Perubahan mAP@5 dari Unigram ke Unigram + Bigram: +6,65 pp. Sebagian kekuatan teks
perlu dibaca bersama temuan deskripsi identik lintas split (Cell 12).

**3. Kinerja weighted score fusion.** Fusion terkunci mencapai mAP@5 0,7052 (validation), yaitu +1,41 pp terhadap
Text Only (Unigram + Bigram) dan +15,30 pp terhadap Visual Only; fusion lebih tinggi pada 5/5 metrik @5.
Secara per query, fusion lebih tinggi pada 4, sama pada 15, dan lebih rendah pada
2 query (Gambar F03). Pada test, selisih mAP@5 terhadap Text Only adalah +0,33 pp
(CI 95% bootstrap −1,81 pp s.d. +2,38 pp), dengan fusion lebih tinggi pada
3/5 metrik. Keuntungan fusion pada test kecil dan/atau tidak konsisten di semua metrik.

**4. Pengaruh pemilihan α.** Kurva α (Gambar F04) menunjukkan puncak mAP@5 validation pada α = 0,3,
dengan rentang mAP@5 sebesar 0,1530 di sepanjang grid. Nilai α dalam 0,01 dari puncak:
0,3; 0,4. Karena α dipilih dari 22 kandidat pada 21 query, skor validation konfigurasi
terpilih cenderung optimistis.

**5. Performa berdasarkan K.** Untuk konfigurasi terkunci, Precision@K validation turun dari 0,7619
(K=1) ke 0,6524 (K=10), sedangkan Recall@K naik ke 0,5149
(Gambar F01). MRR@5 validation 0,8532 menunjukkan item relevan pertama umumnya berada di peringkat awal.

**6. Perbedaan antarkelas.** Kelas terlemah adalah Salam (validation) dan
Salam (test) berdasarkan mAP@5 (Gambar F06). Jumlah query per kelas kecil, sehingga peringkat kelas tidak stabil.

**7. Kesalahan retrieval.** Terdapat 5 kesalahan top-1 pada validation
dan 0 pada test (Gambar F07). Analisis atribusi (Tabel T09b) hanya
menunjukkan keterkaitan dengan ranking modalitas tunggal dan fitur di luar rentang; penyebab pasti tidak dapat disimpulkan dari CSV.

**8. Keterbatasan dataset dan validitas.** Terdapat 44 kandidat near-duplicate lintas split,
4 kelompok deskripsi identik lintas split, identitas spesimen belum dideklarasikan, dan test
sudah pernah diperiksa. Dengan 21 query per partisi, hasil sensitif terhadap beberapa query. Hasil teknis yang lolos audit tidak sama
dengan validitas metodologis.


## Kesimpulan

### A. Kesimpulan teknis (terverifikasi)
1. **Implementasi.** Sistem multimodal retrieval (visual HSV+GLCM+Hu 143 dimensi, teks TF-IDF, weighted score fusion)
   telah berjalan lengkap pada run `20261127T090108Z_c12d4354b8` (status teknis Notebook 06: TECHNICAL_COMPLETE).
   Notebook 07 mereproduksi semua metrik per query, macro, dan per kelas dari ranking tersimpan dengan selisih maksimum
   0.0e+00.
2. **Konfigurasi terpilih** (dari validation, tidak diubah di Notebook 07): TF-IDF Unigram + Bigram,
   α visual = 0,3, bobot teks = 0,7.
3. **Tidak ada penggunaan test untuk seleksi**: selection lock tidak berubah dan menyatakan `selected_using_test = false`.

### B. Klaim ilmiah (dengan kualifikasi)
4. **Apakah fusion meningkatkan hasil?** Ya pada validation untuk mAP@5: fusion lebih tinggi daripada Text Only (+1,41 pp) dan Visual Only (+15,30 pp), serta lebih tinggi pada 5/5 metrik @5 dibanding Text Only. Pada test (PROVISIONAL_AUDIT_UNRESOLVED), ΔmAP@5 terhadap Text Only +0,33 pp dan terhadap Visual Only +20,63 pp; fusion lebih tinggi pada 3/5 metrik dibanding Text Only. Peningkatan atas Text Only pada test kecil atau tidak konsisten, sehingga tidak dapat diklaim sebagai keunggulan yang kuat.
5. **Seberapa besar?** mAP@5 validation: Visual Only 0,5522,
   Text Only (U+B) 0,6911, fusion 0,7052.
   mAP@5 test (provisional): fusion 0,7941. Tidak ada klaim signifikansi statistik:
   interval bootstrap dan uji tanda (Cell 6) bersifat deskriptif.
6. **Status test:** PROVISIONAL_AUDIT_UNRESOLVED. Test **bukan** estimasi generalisasi independen
   (independent = False).

### C. Kelemahan metode dan dataset
- Validation kecil (21 query) dan dipakai untuk memilih α dari 22 kandidat → skor validation terpilih optimistis.
- Kandidat near-duplicate dan deskripsi identik lintas split belum direview; identitas spesimen tidak tersedia.
- Test sudah pernah diperiksa pada run sebelumnya.
- Fitur visual global sensitif terhadap warna; beberapa query berada di luar rentang gallery.
- Kelas terlemah: Salam (validation), Salam (test).

### D. Yang masih perlu diperbaiki
- Menyelesaikan review manual pasangan near-duplicate dan deskripsi identik (file review Notebook 06), lalu bila perlu
  membangun ulang split berbasis spesimen.
- Menambah data atau memakai cross-validation berbasis grup agar seleksi α lebih stabil.
- Menyiapkan test set baru yang belum pernah dilihat untuk estimasi independen.
- Memeriksa gambar dengan fitur visual di luar rentang gallery (warna/segmentasi).


## Lampiran — gambar
- `figures/F01_overall_metrics_by_k.png`
- `figures/F02_method_comparison_at_5.png`
- `figures/F03_fusion_contribution_per_query.png`
- `figures/F04_alpha_sensitivity_validation.png`
- `figures/F05_validation_vs_test_gap.png`
- `figures/F06_per_class_evaluation.png`
- `figures/F07_top1_confusion_matrices.png`
- `figures/F08_example_1_test_leaf_f72f88757c4de8ba.png`
- `figures/F08_example_2_validation_leaf_d12754f94f5ff7fc.png`
- `figures/F08_example_3_test_leaf_1a45e1e109be76ee.png`
- `figures/F09_leakage_robustness.png`
