import pandas as pd

dados = {
    "Dia": [f"D{i}" for i in range(1, 15)],
    "Aspecto": ["Sol", "Sol", "Nuvens", "Chuva", "Chuva", "Chuva", "Nuvens", "Sol",
                "Sol", "Chuva", "Sol", "Nuvens", "Nuvens", "Chuva"],
    "Temp.": ["Quente", "Quente", "Quente", "Ameno", "Fresco", "Fresco", "Fresco", "Ameno",
              "Fresco", "Ameno", "Ameno", "Ameno", "Quente", "Ameno"],
    "Humidade": ["Elevada", "Elevada", "Elevada", "Elevada", "Normal", "Normal", "Normal", "Elevada",
                 "Normal", "Normal", "Normal", "Elevada", "Normal", "Elevada"],
    "Vento": ["Fraco", "Forte", "Fraco", "Fraco", "Fraco", "Forte", "Fraco", "Fraco",
              "Fraco", "Forte", "Forte", "Forte", "Fraco", "Forte"],
    "Jogar Tênis": ["Não", "Não", "Sim", "Sim", "Sim", "Não", "Sim", "Não",
                    "Sim", "Sim", "Sim", "Sim", "Sim", "Não"]
}

df = pd.DataFrame(dados)
print(df.head())

def effect_encode(column):
    dummies = pd.get_dummies(column)
    last_col = dummies.columns[-1]
    dummies[last_col] = -1 * dummies.drop(columns=last_col).sum(axis=1)
    return dummies

cols_cat = ["Aspecto", "Temp.", "Humidade", "Vento"]
df_effect = pd.DataFrame()

for col in cols_cat:
    encoded = effect_encode(df[col])
    encoded.columns = [f"{col}_{c}" for c in encoded.columns]
    df_effect = pd.concat([df_effect, encoded], axis=1)

df_effect["Jogar Tênis"] = df["Jogar Tênis"]
print("\nEffect Encoding:")
print(df_effect.head())