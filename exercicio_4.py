import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

base_path = "aclImdb/train"
pos_path = os.path.join(base_path, "pos")
neg_path = os.path.join(base_path, "neg")

def carregar_resenhas(caminho, rotulo):
    textos = []
    for nome_arquivo in os.listdir(caminho)[:2500]:
        with open(os.path.join(caminho, nome_arquivo), encoding='utf-8') as f:
            textos.append(f.read())
    return pd.DataFrame({"review": textos, "label": rotulo})

df_pos = carregar_resenhas(pos_path, 1)
df_neg = carregar_resenhas(neg_path, 0)
df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df['review'])

print("Matriz TF-IDF criada com shape:", X_tfidf.shape)

features = vectorizer.get_feature_names_out()

X_dense = X_tfidf.toarray()

max_tfidf_por_palavra = X_dense.max(axis=0)

df_tfidf = pd.DataFrame({
    'palavra': features,
    'max_tfidf': max_tfidf_por_palavra
})

top_10 = df_tfidf.sort_values(by='max_tfidf', ascending=False).head(10)
bottom_10 = df_tfidf.sort_values(by='max_tfidf', ascending=True).head(10)

print("\nTop 10 palavras com maior TF-IDF:")
print(top_10)

print("\nBottom 10 palavras com menor TF-IDF:")
print(bottom_10)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['label'], test_size=0.2, random_state=42)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo de Regressão Logística: {acuracia:.4f}")

coeficientes = modelo.coef_[0]
features = vectorizer.get_feature_names_out()

df_coef = pd.DataFrame({
    'palavra': features,
    'coeficiente': coeficientes
})

top_40 = df_coef.sort_values(by='coeficiente', ascending=False).head(40)
bottom_40 = df_coef.sort_values(by='coeficiente', ascending=True).head(40)

plt.figure(figsize=(12, 6))
plt.barh(top_40['palavra'], top_40['coeficiente'])
plt.xlabel("Coeficiente")
plt.title("🔝 40 palavras mais indicativas de sentimento positivo")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.barh(bottom_40['palavra'], bottom_40['coeficiente'], color='red')
plt.xlabel("Coeficiente")
plt.title("🔻 40 palavras mais indicativas de sentimento negativo")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
