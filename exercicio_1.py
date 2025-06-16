import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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
