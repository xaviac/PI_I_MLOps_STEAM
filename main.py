import pandas as pd
import joblib
from fastapi import FastAPI

group_by_year_genres = pd.read_parquet('https://github.com/xaviac/storage__PI_MLOp/raw/main/data/functions/group_by_year_genres.parquet.gz')
group_by_user_genres_year = pd.read_parquet('https://github.com/xaviac/storage__PI_MLOp/raw/main/data/functions/group_by_user_genres_year.parquet.gz')
union_ur_sg = pd.read_parquet('https://github.com/xaviac/storage__PI_MLOp/raw/main/data/functions/union_ur_sg.parquet.gz')
df_model_fit = pd.read_parquet('https://github.com/xaviac/storage__PI_MLOp/raw/main/data/model/df_model_fit.parquet.gz')

with open('./model/cosine_similarity.pkl', 'rb') as file:
    modelo = joblib.load(file)

app = FastAPI(title="Steam API",
              description="API del Sistema de recomendación de la plataforma Steam",
              version="0.1")


@app.get("/")
async def root():
    return {"Visit this url": "https://mlops-steam-api.onrender.com/docs"}

@app.get("/PlayTimeGenre/{genero}")
async def PlayTimeGenre(genero: str):
    """Función que devuelve el tiempo total de juego para un género dado."""
     # Filtramos el dataframe para el generes
    df_genres = group_by_year_genres[group_by_year_genres['genres'].str.contains(genero, case=False, na=False)]

    # Agrupamos release_year y sumamos playtime_forever_hours
    total_by_year = df_genres.groupby('release_year')['playtime_forever_hours'].sum()
    # Se encuentra el año con más horas jugadas
    top_year = total_by_year.idxmax()

    result = {f'Año de lanzamiento con más horas jugadas para {genero}: {top_year}'}

    return result


@app.get("/UserForGenre/{genero}")
async def UserForGenre(genero: str):
    """Función que devuelve el usuario con más horas jugadas para un género dado."""

    # Filtrar el DataFrame por el género dado
    generes = group_by_user_genres_year[group_by_user_genres_year['genres'].str.contains(genero)] 
    #Agrupo por usuario y sumo cantidad de horas jugadas
    hours = generes.groupby('user_id')['playtime_forever_hours'].sum().reset_index()

    # Encontrar el usuario con más horas jugadas para ese género
    playtime_by_user = hours.loc[hours['playtime_forever_hours'].idxmax()]['user_id']


    # Filtrar el DataFrame por el usuario con más horas jugadas en ese género
    top_user = generes[generes['user_id'] == playtime_by_user]
    total_hours_year = top_user.groupby('release_year')['playtime_forever_hours'].sum().reset_index()

    user_hours_years = [{'Año': int(row['release_year']), 'Horas': int(row['playtime_forever_hours'])} for _, row in total_hours_year.iterrows()]

    result = {f'Usuario con más horas jugadas para {genero}': playtime_by_user, 'Horas jugadas': user_hours_years}
    
    return result


@app.get("/UsersRecommend/{year}")
async def UsersRecommend(year: int):
    """Función que devuelve los 3 juegos más recomendados para un año dado."""
    # Filtramos el DataFrame donde la columna 'release_year' es igual a year, la columna 'recommend' es True y la columna 'sentiment_analysis' tiene valores 1 o 2. 
    filter = union_ur_sg['title'][(union_ur_sg['release_year'] == year) & (union_ur_sg['recommend'] == True) & (union_ur_sg['sentiment_analysis'].isin([1, 2]))].value_counts().reset_index().head(3)
    
    return [{f'Puesto {i + 1}: {row['title']}'} for i, row in filter.iterrows()]


@app.get("/UsersNotRecommend/{year}")
async def UsersNotRecommend(year: int):
    """Función que devuelve los 3 juegos menos recomendados para un año dado."""
    # Se filtra las filas del DataFrame donde la columna 'release_year' es igual a year, la columna 'recommend' es False y la columna 'sentiment_analysis' con valor 0
    filter = union_ur_sg['title'][(union_ur_sg['release_year'] == year) & (union_ur_sg['recommend'] == False) & (union_ur_sg['sentiment_analysis']==0)].value_counts().reset_index().head(3)

    return [{f'Puesto {i + 1}: {row['title']}'} for i, row in filter.iterrows()]


@app.get("/sentiment_analysis/{year}")
async def sentiment_analysis(year: int):
    """Función que devuelve la cantidad de comentarios positivos, negativos y neutrales para un año dado."""
    # Filtrar el DataFrame para el año
    filter = union_ur_sg[union_ur_sg['release_year'] == year]
    #Cuenta los comentarios positivos
    positives = filter[filter['sentiment_analysis']==2]['sentiment_analysis'].count()
    # Cuenta los comentarios negativos
    negatives = filter[filter['sentiment_analysis']==0]['sentiment_analysis'].count()
    # Cuenta los comentarios neutrales
    neutrals = filter[filter['sentiment_analysis']==1]['sentiment_analysis'].count()
    # Devolver conteos en un diccionario

    result = {'Negative': int(negatives), 'Positive': int(positives), 'Neutral': int(neutrals)}

    return result


@app.get("/recomendacion_juego/{item_id}")
async def recomendacion_juego(item_id: int):
    
    if item_id not in df_model_fit['id'].tolist():
       return {'Respuesta':'No se encontraron resultados para el item_id: {}'.format(item_id)}
    
    def get_recommendations(idx, cosine_sim=modelo):
     if idx >= len(cosine_sim):
          return 'No se encontraron resultados para el item_id: {}'.format(item_id)

     sim_scores = list(enumerate(cosine_sim[idx]))
     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
     sim_scores = sim_scores[1:6]
     game_indices = [i[0] for i in sim_scores]
     return df_model_fit['title'].iloc[game_indices].tolist()
    
    #Obtener el índice del item_id
    recommendations = get_recommendations(item_id)
    
    return {"Recomendaciones": recommendations}