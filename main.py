import pandas as pd
import joblib
from fastapi import FastAPI

df_sg = pd.read_parquet('https://github.com/xaviac/storage__PI_MLOp/raw/main/data/clean/steam_games.parquet.gz')
group_by_year_genres = pd.read_parquet('https://github.com/xaviac/storage__PI_MLOp/raw/main/data/functions/group_by_year_genres.parquet.gz')
items_reviews_users = pd.read_parquet('https://github.com/xaviac/storage__PI_MLOp/raw/main/data/functions/items_reviews_users.parquet.gz')
group_by_user_genres_year = pd.read_parquet('https://github.com/xaviac/storage__PI_MLOp/raw/main/data/functions/group_by_user_genres_year.parquet.gz')
union_ur_sg = pd.read_parquet('https://github.com/xaviac/storage__PI_MLOp/raw/main/data/functions/union_ur_sg.parquet.gz')
df_model_fit = pd.read_parquet('https://github.com/xaviac/storage__PI_MLOp/raw/main/data/model/df_model_fit.parquet.gz')

with open('./model/cosine_similarity.pkl', 'rb') as file:
    modelo = joblib.load(file)

app = FastAPI()


@app.get("/")
async def root():
    return {"Visit this url": "https://mlops-steam-api.onrender.com/docs"}

@app.get("/developer/{desarrollador}")
async def developer(desarrollador: str):
    """
    Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora
    params:
    desarrollador: str
    """
    # Verifiquemos que se ha introducido un desarrollador que existe en el dataset
    if desarrollador.title() not in list(df_sg['developer'].str.title()):
        return "Desarrollador no encontrado, por favor, inténtelo de nuevo." 
    
    group_by_year = df_sg[df_sg['developer'].str.title() == desarrollador.title()].groupby('release_year')['id'].count().reset_index()


    # Conteo de 'Free to Play' por año
    free_content = df_sg[(df_sg['developer'] == desarrollador) & (df_sg['price'] == 0)].groupby('release_year')['id'].count().reset_index()
    free_content.rename(columns={'id': 'Free to Play'}, inplace=True)
    
    # Unión de group_by_year y free_content
    merged_data = pd.merge(group_by_year, free_content, on='release_year', how='left')
    
    # Porcentaje del contenido gratis por año
    percent_free = round(merged_data['Free to Play'] / merged_data['id'] * 100, 0)

    list_percent_free = percent_free.fillna(0).tolist()

    # Convertir a cadena y formatear la presentación
    formatted_list = [f'{int(num)}%' for num in list_percent_free]
    
    # Creo el DataFrame final
    resultado = {
        'Año': merged_data['release_year'].tolist(),
        'Cantidad de Items': merged_data['id'].tolist(),
        'Contenido Free': formatted_list
    }
    
    return resultado

@app.get("/userdata/{User_id}")
async def userdata(User_id:str):
    """
    Retorna el dinero gastado, el porcentaje de recomendación y la cantidad de items comprados por el usuario
    params:
    User_id: str
    """
    # Se genera un dataframe para el User_id ingresado de donde se leerán los datos.
    user = items_reviews_users[items_reviews_users['user_id'] == User_id]
    if user.empty:
        return "Usuario no encontrado"
    
    return {
        'Usuario': str(user['user_id'].iloc[0]),
        'Dinero gastado': f'{str(user["price"].iloc[0])} USD',
        '% de recomendación': f'{str(user['percent_recommend'].iloc[0])}%',
        'Cantidad de items': str(int(user['total_items'].iloc[0]))
        }

@app.get("/UserForGenre/{genero}")
async def UserForGenre(genero: str):
    """
    Función que devuelve el usuario con más horas jugadas para un género dado.
    params:
    genero: str
    """

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
    filter = union_ur_sg['title'][(union_ur_sg['release_year'] == year) & (union_ur_sg['recommend'] is True) & (union_ur_sg['sentiment_analysis'].isin([1, 2]))].value_counts().reset_index().head(3)

    result = [{'Puesto {}: {}'.format(i + 1, row['title'])} for i, row in filter.iterrows()]
    
    return result


@app.get("/UsersNotRecommend/{year}")
async def UsersNotRecommend(year: int):
    """Función que devuelve los 3 juegos menos recomendados para un año dado."""
    # Se filtra las filas del DataFrame donde la columna 'release_year' es igual a year, la columna 'recommend' es False y la columna 'sentiment_analysis' con valor 0
    filter = union_ur_sg['title'][(union_ur_sg['release_year'] == year) & (union_ur_sg['recommend'] is False) & (union_ur_sg['sentiment_analysis']==0)].value_counts().reset_index().head(3)

    result = [{'Puesto {}: {}'.format(i + 1, row['title'])} for i, row in filter.iterrows()]
    
    return result


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