import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Veri Yükleme (Örnek Sentetik Veri Seti - Kaggle "Sleep Health and Lifestyle" baz alınmıştır)
# Not: Orijinal veri setini aynı dizine 'sleep_health_data.csv' olarak eklemelisin.
df = pd.read_csv('sleep_health_data.csv')

# Bağımsız değişkenler (Features) ve Bağımlı değişken (Target)
features = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Daily Steps', 'Heart Rate']
X = df[features]
y = df['Stress Level']

# 2. Veri Ön İşleme (Z-Score Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA (Principal Component Analysis) ile Boyut İndirgeme
# 5 boyuttan 2 boyuta (Sleep-Recovery ve Physical Activity eksenleri) indirgiyoruz.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Açıklanan Varyans Oranı (Explained Variance Ratio): {sum(pca.explained_variance_ratio_):.2f}")

# 4. Veriyi Train/Test olarak %80'e %20 ayırma
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.20, random_state=42)

# 5. Lineer Regresyon Modeli Oluşturma ve Eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Model Değerlendirmesi
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
