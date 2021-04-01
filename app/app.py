  
import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from plotly.tools import mpl_to_plotly
from matplotlib import pyplot as plt
import plotly.express as px

import yaml
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from constants import CLASSES

# Configuration yaml

# Function to load yaml configuration file
def load_config(parameters):
    with open(os.path.join('C:/Users/chahinez/Documents/hands-on-2021/app/app.yaml', parameters)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config('C:/Users/chahinez/Documents/hands-on-2021/app/app.yaml')

#Ouverture base 

df = pd.read_csv(config['data1'])

IMAGE_WIDTH = config['IMAGE_WIDTH']
IMAGE_HEIGHT = IMAGE_WIDTH

# Load DNN model
classifier = tf.keras.models.load_model(config['model2'])

def classify_image(image, model, image_box=None):
  """Classify image by model
  Parameters
  ----------
  content: image content
  model: tf/keras classifier
  Returns
  -------
  class id returned by model classifier
  """
  images_list = []
  image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box)
                                        # box argument clips image to (x1, y1, x2, y2)
  image = np.array(image)
  images_list.append(image)
  
  return model.predict_classes(np.array(images_list))

# Classify for probability
def classify_image2(image, model, image_box=None):
  """Classify image by model
  Parameters
  ----------
  content: image content
  model: tf/keras classifier
  Returns
  -------
  class id returned by model classifier
  """
  images_list = []
  image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box)
                                        # box argument clips image to (x1, y1, x2, y2)
  image = np.array(image)
  images_list.append(image)
  
  return model.predict(np.array(images_list))

def plot_value_array(image, model):
    predictions1= classify_image2(image, model)[0]
    #predictions1 = predictions1.flatten()
    
    fig, ax = plt.subplots(figsize=(20, 6), dpi=80)
    ax.set_xticks(range(43))
    ax.bar(range(43), predictions1, color="#777777")
    plotly_fig = mpl_to_plotly(fig)
    
app = dash.Dash('Traffic Signs Recognition') #, external_stylesheets=dbc.themes.BOOTSTRAP)


pre_style = {
    'whiteSpace': 'pre-wrap',
    'wordBreak': 'break-all',
    'whiteSpace': 'normal'
}

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


# Define application layout
app.layout = html.Div([
    dbc.Row(dbc.Col(html.H1(
        children='Traffic Signs Recognition',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ))),
    html.Div(children='A web application for Python to classify images', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    html.Hr(),
    html.Hr(),
    html.H6('Le but de cette application est de classer des panneaux de signalisation. Veuillez donc essayer avec une image et constater les performances de prédiction.'),
    
    dcc.Upload(
        id='bouton-chargement',
        children=html.Div([
            'Cliquer-déposer ou ',
                    html.A('sélectionner une image')
        ]),
        style={
            'width': '50%',
            'height': '80px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '60px',
            'color': colors['text'],
            'backgroundColor': colors['background']
        }
    ),
    html.Div(id='mon-image'),
    #dcc.Input(id='mon-champ-texte', value='valeur initiale', type='text'),
    #html.Div(id='ma-zone-resultat'),
    html.H6('Vous avez apprecié découvrir le résultat de prédiction du modèle? Deroulez le folder pour connaitre ses capacités prédictives sur l_image que vous souhaitez.'),
    html.Div([
        dcc.Graph(id = 'graph-with-slider'),
        dcc.Slider(
            id = 'nbimage-slider',
            min = df['nbimage'].min(),
            max = df['nbimage'].max(),
            value = df['nbimage'].min(),
            marks= {str(nbimage): str(nbimage) for nbimage in df['nbimage'].unique()},
            step = None
        ),
        html.Hr(),
        html.Hr(),
        html.H5('Image choisie : '),
        html.Div(id='display-selected-values'),
        html.Hr(),
        html.Hr()
    ])
])
# call back for image
@app.callback(Output('mon-image', 'children'),
              [Input('bouton-chargement', 'contents')])
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        if 'image' in content_type:
            image = Image.open(io.BytesIO(base64.b64decode(content_string)))
            predicted_class = classify_image(image, classifier)[0]
            return html.Div([
                html.Hr(),
                html.Img(src=contents),
                html.H3('Classe prédite : {}'.format(CLASSES[predicted_class])),
                html.Hr(),
                #html.Div('Raw Content'),
                #html.Pre(contents, style=pre_style)
            ])
        else:
            try:
                # Décodage de l'image transmise en base 64 (cas des fichiers ppm)
                # fichier base 64 --> image PIL
                image = Image.open(io.BytesIO(base64.b64decode(content_string)))
                # image PIL --> conversion PNG --> buffer mémoire 
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                # buffer mémoire --> image base 64
                buffer.seek(0)
                img_bytes = buffer.read()
                content_string = base64.b64encode(img_bytes).decode('ascii')
                # Appel du modèle de classification
                predicted_class = classify_image(image, classifier)[0]
                predictedess = classify_image2(image, classifier)[0]
                # Affichage de l'image
                return html.Div([
                    html.Hr(),
                    html.Img(src='data:image/png;base64,' + content_string),
                    html.H3('Classe prédite : {}'.format(CLASSES[predicted_class]),
                            style={
                                'textAlign': 'center',
                                'color': colors['text']
                            }
                           ),
                    html.H4('Ce qui correspond à la classe {}'.format([np.argmax(predictedess)]),
                            style={
                                'textAlign': 'center'
                            }
                           ),
                    html.H4('ceci avec une confiance de {} %'.format(round(100*(np.max(predictedess)),2)),
                            style={
                                'textAlign': 'center'
                            }
                           ),
                    html.Hr()
                ])
            except:
                return html.Div([
                    html.Hr(),
                    html.Div('Uniquement des images svp : {}'.format(content_type)),
                    html.Hr(),                
                    html.Div('Raw Content'),
                    html.Pre(contents, style=pre_style)
                ])
# colback for bar chart            
@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('nbimage-slider', 'value'))
def update_figure(selected_nbimage):
    filtered_df = df[df.nbimage == selected_nbimage]
    fig = px.bar(filtered_df, x="classe", y="confiance")
    fig.update_layout(transition_duration=500)

    return fig

@app.callback(
    Output('display-selected-values', 'children'),
    Input('nbimage-slider', 'value'))
def set_display_children(selected_nbimage):
    return 'L_image choisie est l_image numero {}'.format(
        selected_nbimage)


# Start the application
if __name__ == '__main__':
    app.run_server(debug=True)
