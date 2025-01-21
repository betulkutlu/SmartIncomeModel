import numpy as np
import matplotlib.pyplot as plt

# Rastgele veri seti oluşturma
np.random.seed(63)
X = np.random.randint(9, 17, 130)
noise = np.random.normal(0, 10, 130)
Y = 13 * X + 45 + noise  # m=13, b=45 sabit değerler

# Başlangıç değerlerini rastgele seçiyoruz
m = 0  # Başlangıç eğimi
b = 0  # Başlangıç kesim noktası
learning_rate = 0.001  # Öğrenme oranı
epochs = 1000  # Iterasyon sayısı

# Hata fonksiyonu (MSE) tanımlaması
def compute_error(X, Y, m, b):
    n = len(X)
    Y_pred = m * X + b
    error = (1/n) * np.sum((Y - Y_pred)**2)  # MSE (Mean Squared Error)
    return error

# Gradyan inişi algoritması.
for epoch in range(epochs):
    # Modelin tahminlerini hesapla
    Y_pred = m * X + b
    
    # m ve b için türevler (gradientler)
    dm = -(2/len(X)) * np.sum(X * (Y - Y_pred))
    db = -(2/len(X)) * np.sum(Y - Y_pred)
    
    # m ve b değerlerini güncelle
    m -= learning_rate * dm
    b -= learning_rate * db

    # Hata fonksiyonunu yazdır (her 100. adımda)
    if epoch % 100 == 0:
        error = compute_error(X, Y, m, b)
        print(f"Epoch {epoch}, Error {error}, m {m}, b {b}")

# Eğitim tamamlandıktan sonra tahmin yap
new_X = np.array([10, 12, 15]) 
predictions = m * new_X + b
print("Yeni X değerleri için tahminler:", predictions)

# Sonuç grafikleri
plt.scatter(X, Y, color='blue', label='Veri')
plt.plot(X, m * X + b, color='red', label='Lineer Regresyon Doğrusu')
plt.scatter(new_X, predictions, color='green', label='Tahminler')  # Yeni tahminleri işaretle
plt.title('Çalışma Süresi vs Gelir - Lineer Regresyon')
plt.xlabel('Çalışma Süresi (Saat)')
plt.ylabel('Gelir')
plt.legend()
plt.show()
