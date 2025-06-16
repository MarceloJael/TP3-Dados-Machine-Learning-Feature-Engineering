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

df_dummy = pd.get_dummies(df.drop(columns=["Dia", "Jogar Tênis"]), drop_first=True)
df_dummy["Jogar Tênis"] = df["Jogar Tênis"]
print("\nDummy Encoding:")
print(df_dummy.head())