import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - importar arquivo csv
df = pd.read_csv('medical_examination.csv')

# 2 - criar nova coluna 'overweight' 
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)


# 3 - normalizar colunas de cholesterol e gluc
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4 
def draw_cat_plot():
    # 5 - remodelar as colunas
    df_cat = df.melt(id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], var_name='variable', value_name='value')


    # 6 - agrupar a base de dados pelas features 'cardio','variable' e 'value'
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')
    df_cat.rename(columns={'count': 'total'}, inplace=True)

    # 7    
    # Criar o gráfico categórico usando catplot
    cat_plot = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar', height=5, aspect=1.5)


    # 8
    fig = cat_plot.fig
    # 9 - Salvando figura
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 1. limpar os dados de acordo com os limites estabelecidos
    df_heat = df[
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 2. calcular a matriz de correlação 
    corr = df_heat.corr()

    # 3. gerar uma máscara para o triângulo superior da matriz de correlação
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 4. configurar a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))

    # 5. desenhar o heatmap usando seaborn
    sns.heatmap(
        corr,                      
        mask=mask,                 
        annot=True,                
        fmt=".1f",                 
        cmap='coolwarm',         
        vmax=0.3,                   
        center=0,                   
        square=True,               
        linewidths=0.5,             
        cbar_kws={"shrink": 0.5},   
        ax=ax                       
    )

    # 6. salvar a figura
    fig.savefig('heatmap.png')
    
    return fig