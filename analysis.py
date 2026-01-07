import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = {
    'Provinsi': ['Jawa Tengah', 'Jawa Barat', 'Jawa Timur', 'Sumatera Utara', 'Riau', 'Sulawesi Selatan', 'Aceh'],
    'Banjir': [10, 12, 8, 10, 3, 1, 2],
    'Cuaca_Ekstrem': [25, 20, 12, 2, 0, 8, 0],
    'Karhutla': [0, 0, 0, 0, 8, 1, 7],
    'Longsor': [2, 2, 0, 2, 0, 0, 0]
}
df = pd.DataFrame(data)


features = df[['Banjir', 'Cuaca_Ekstrem', 'Karhutla', 'Longsor']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)


def interpret_cluster(row):
    if row['Cuaca_Ekstrem'] > 15:
        return "Zona Rawan Cuaca Ekstrem", "Perkuat struktur atap & pangkas pohon tua (Penyebab 81.6% rumah rusak)."
    elif row['Karhutla'] > 5:
        return "Zona Rawan Karhutla", "Monitoring lahan gambut & siapkan embung air (Hidrometeorologi Kering)."
    elif row['Banjir'] >= 8:
        return "Zona Rawan Banjir/Longsor", "Normalisasi sungai, perbaikan tanggul, & cek retakan tanah di lereng."
    else:
        return "Zona Risiko Menengah", "Edukasi tas siaga bencana & pemantauan info cuaca rutin."

df[['Nama_Cluster', 'Rekomendasi']] = df.apply(
    lambda x: pd.Series(interpret_cluster(x)), axis=1
)


plt.figure(figsize=(12, 7))
sns.scatterplot(data=df, x='Banjir', y='Cuaca_Ekstrem', hue='Nama_Cluster', s=300, palette='Set1')

for i in range(df.shape[0]):
    plt.text(df.Banjir[i]+0.3, df.Cuaca_Ekstrem[i], df.Provinsi[i], weight='bold')

plt.title('Cluster Wilayah Rawan Bencana Indonesia (Data Oktober 2025)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


print("\n=== HASIL ANALISIS PRIORITAS MITIGASI ===")
print(df[['Provinsi', 'Nama_Cluster', 'Rekomendasi']].to_string(index=False))
def simpan_tabel_ke_gambar(df):

    df_plot = df[['Provinsi', 'Nama_Cluster', 'Rekomendasi']].copy()
    df_plot['Rekomendasi'] = df_plot['Rekomendasi'].apply(lambda x: "\n".join(textwrap.wrap(x, width=50)))

    plt.figure(figsize=(14, 8)) 
    plt.axis('off')

    tabel = plt.table(
        cellText=df_plot.values, 
        colLabels=['Provinsi', 'Kategori Risiko', 'Rekomendasi Prioritas'], 
        cellLoc='left', 
        loc='center'
    )

    tabel.auto_set_font_size(False)
    tabel.set_fontsize(10)

    tabel.scale(1.2, 4.5) 

    for (row, col), cell in tabel.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        
    plt.title('Daftar Rekomendasi Mitigasi Berdasarkan Analisis Klaster', pad=30, weight='bold', fontsize=14)
    
    plt.savefig('tabel_rekomendasi.png', bbox_inches='tight', dpi=300)
    print("Gambar tabel rapi berhasil disimpan: tabel_rekomendasi.png")

simpan_tabel_ke_gambar(df)