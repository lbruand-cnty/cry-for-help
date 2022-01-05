import os
import shutil
import pandas as pd
from datetime import datetime
from flask import Flask, request
from flask_cors import cross_origin
from flask_restful import Api
import ml_api
from srcs import utils

CONFIG = utils.load_yaml('./config.yaml')
API_ENDPOINTS = CONFIG['API_ENDPOINTS']
PROJECT_DIR = CONFIG['PROJECT_DIR']

os.makedirs(PROJECT_DIR, exist_ok=True)
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)

mlapi = None

@app.route(f'{API_ENDPOINTS["ADD_DATA"]}/<project_name>', methods=['PUT'])
@cross_origin()
def add_text_data(project_name: str):
    """
    Add texts to be labelled. This api expects json data as follows:
    {
        'texts': List[str],
    }

    Args:
        project_name (str): Project name.
    """
    new_data = request.get_json()
    new_data['verified'] = ['0'] * len(new_data['texts'])
    new_data['label'] = None
    new_data['queue'] = None
    df = pd.DataFrame(new_data)
    df.to_csv(os.path.join(PROJECT_DIR, project_name, 'data.csv'), index=False)
    return {'success': True}, 200, {'ContentType': 'application/json'}


@app.route(f'{API_ENDPOINTS["DOWNLOAD_DATA"]}/<project_name>/<all_or_labeled>', methods=['GET'])
@cross_origin()
def download_data(project_name: str, all_or_labeled: str):
    """
    Download csv containing all data or just labeled data.

    Args:
        project_name (str): Project name.
        all_or_labeled (str): Specify 'labeled' to download labeled data else
                              all data will be downloaded.

    Returns:
        {
            'text': Text data, List[str],
            'verified': Verification datetime, List[str],
            'label': Comma separated labels, List[str],
        }
    """
    df = pd.read_csv(os.path.join(PROJECT_DIR, project_name, 'data.csv'))
    if all_or_labeled == 'labeled':
        df = df[df['verified'] != '0']
    # process the labels
    df['label'] = df['label'].apply(lambda x: str(x).replace(':sep:', ', '))
    text = df.texts.to_list()
    verified = df.verified.to_list()
    queue = df.queue.to_list()
    label = df.label.to_list()
    return {
        'text': text,
        'verified': verified,
        'queue': queue,
        'label': label,
    }


@app.route(f'{API_ENDPOINTS["GET_DATA"]}/<project_name>/<int:current_page>', methods=['GET'])
@cross_origin()
def get_data(project_name: str, current_page: int):
    """
    Get and return data, verification datetime and label in a dictionary.

    Args:
        project_name (str): Project name.
        current_page (int): Current page index.

    Returns:
        {
            'total': Total number of data in current project, int,
            'text': Text data of the current page index, str,
            'verified': Verification datetime of the current data, str,
            'label': ":sep:" separated labels, str,
        }
    """
    try:
        df = pd.read_csv(os.path.join(PROJECT_DIR, project_name, 'data.csv'))
        total = len(df)  # TODO : This should be a project wide info
        current_page = min(total - 1, current_page)
        text = df.texts.iloc[current_page]
        verified = str(df.verified.iloc[current_page])
        queue = str(df.queue.iloc[current_page])
        label = str(df.label.iloc[current_page])
        label = [] if label == 'nan' else label.split(':sep:')
    except FileNotFoundError:
        total = 0
        text = None
        verified = None
        queue = None
        label = None
    return {
        'total': total,
        'text': text,
        'verified': verified,
        'queue': queue,
        'label': label,
    }

@app.route(f'{API_ENDPOINTS["GET_PROJECT_INFO"]}/<project_name>', methods=['GET'])
@cross_origin()
def get_project_info(project_name: str):
    """
    Get and return project information in a dictionary.

    Args:
        project_name (str): Project name.

    Returns:
        {
            'project': Current project name, str,
            'createDate': Project creation datetime, str,
            'description': Project description, str,
            'label': List of labels defined, List[str],
            'label_shortcut': List of label shortcuts defined, List[str]
            'progress': Number of labeled data in current project, str,
        }
    """
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'projects.csv'))
    selected_df = df.loc[df.project == project_name].reset_index()
    label = selected_df.label[0]
    label = [] if str(label) == 'nan' else label.split(':sep:')
    label_shortcut = selected_df.label_shortcut[0]
    label_shortcut = [] if str(label_shortcut) == 'nan' else label_shortcut.split(':sep:')
    # get proportion of labeled data
    try:
        df = pd.read_csv(os.path.join(PROJECT_DIR, project_name, 'data.csv'))
        progress = str(len(df) - df['verified'].value_counts()['0'])
    except FileNotFoundError:  # if no data been added
        progress = None
    except KeyError:  # if no labeled data
        progress = '0'
    return {
        'project': project_name,
        'createDate': selected_df.createDate[0],
        'description': selected_df.description[0],
        'label': label,
        'label_shortcut': label_shortcut,
        'progress': progress,
    }


@app.route(API_ENDPOINTS['LOAD_PROJECTS'], methods=['GET'])
@cross_origin()
def get_all_projects():
    """
    Get and return list of projects.

    Returns:
        {
            'projects': List of project names, List[str]
        }
    """
    try:
        df = pd.read_csv(os.path.join(PROJECT_DIR, 'projects.csv'))
        projects = df.project.to_list()
    except FileNotFoundError:
        projects = []

    return {'projects': projects}


@app.route(f'{API_ENDPOINTS["CREATE_PROJECT"]}/<project_name>', methods=['PUT'])
@cross_origin()
def create_project(project_name: str):
    """
    Create a new project.

    Args:
        project_name (str): Project name.
    """
    # create folder
    os.makedirs(os.path.join(PROJECT_DIR, project_name), exist_ok=True)
    # add project info
    to_append = {
        'project': [project_name],
        'createDate': [str(datetime.now()).split('.')[0][:-3]],
        'description': ['Add description at here.'],  # default description
        'label': None,  # default label
        'label_shortcut': None
    }
    df_to_append = pd.DataFrame(to_append)
    try:
        df = pd.read_csv(os.path.join(PROJECT_DIR, 'projects.csv'))
        df = df.append(df_to_append, ignore_index=True)
    except FileNotFoundError:
        df = df_to_append

    df.to_csv(os.path.join(PROJECT_DIR, 'projects.csv'), index=False)
    return {'success': True}, 200, {'ContentType': 'application/json'}


@app.route(f'{API_ENDPOINTS["DELETE_PROJECT"]}/<project_name>', methods=['DELETE'])
@cross_origin()
def delete_project(project_name):
    """
    Delete an existing project.

    Args:
        project_name (str): Project name.
    """
    # delete folder
    shutil.rmtree(os.path.join(PROJECT_DIR, project_name))
    # delete project info
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'projects.csv'))
    df = df[df['project'] != project_name]
    df.to_csv(os.path.join(PROJECT_DIR, 'projects.csv'), index=False)
    return {'success': True}, 200, {'ContentType': 'application/json'}


@app.route(f'{API_ENDPOINTS["UPDATE_LABEL_DATA"]}/<project_name>/<int:current_page>', methods=['PUT'])
@cross_origin()
def update_label_data(project_name: str, current_page: int):
    """
    Update the labeled data. This api expects json data as follows:
    {
        'new_labels': List of labels, List[str],
        'verified': Verification datetime, str,
    }

    Args:
        project_name (str): Project name.
        current_page (int): Current page index.
    """
    new_labels = request.get_json()['new_labels']
    verified = request.get_json()['verified']
    queue = request.get_json()['queue']
    df = pd.read_csv(os.path.join(PROJECT_DIR, project_name, 'data.csv'))
    df.verified.iloc[current_page] = verified
    df.queue.iloc[current_page] = queue
    df.label.iloc[current_page] = ':sep:'.join(new_labels)
    df.to_csv(os.path.join(PROJECT_DIR, project_name, 'data.csv'), index=False)
    return {'success': True}, 200, {'ContentType': 'application/json'}

@app.route(f'{API_ENDPOINTS["SAMPLE_DATA"]}/<project_name>', methods=['POST'])
@cross_origin()
def sample_data(project_name):
    df = pd.read_csv(os.path.join(PROJECT_DIR, project_name, 'data.csv'))
    df_train = df[df.queue == "train"] # TODO : Hardcoded values should be params.
    df_test = df[df.queue == "test"]
    df_unlabeled = df[ ~df.queue.isin(["train", "test"]) ]
    print(f" len(df_train) = {len(df_train)}")
    print(f" len(df_test) = {len(df_test)}")
    print(f" len(df_unlabeled) = {len(df_unlabeled)}")

    df = pd.concat(
        [
            *list(sample_unlabeled_data(df_unlabeled, df_train, df_test)),  #  TODO : This needs to be done using the model + sampling + oultliers.
            df_train,
            df_test
        ])
    df.to_csv(os.path.join(PROJECT_DIR, project_name, 'data.csv'), index=False)
    return {'success': True}, 200, {'ContentType': 'application/json'}


def sample_unlabeled_data(df_unlabeled, df_train, df_test): # TODO : move this ml_api.
    global mlapi
    print(" == sample_unlabeled_data ==")
    if mlapi is None:
        mlapi = ml_api.MLApi()
        # TODO : Hardcoded ... make it better.
        feature_index = {'hello': 0, 'Programmation': 1, 'et': 2, 'développement': 3, 'informatique': 4}
        mlapi = ml_api.MLApi(labels_index={"GRAIT": 0, "GRAMC": 1},
                             num_labels=2,
                             vocab_size=len(feature_index))
        mlapi.feature_index = feature_index
    # TODO : Train every n labels.
    mlapi.train_model(training_data=df_train,
                      test_data=df_test,)
    df = df_unlabeled.sample(frac=1).reset_index(drop=True)
    df_result = mlapi.get_low_conf_unlabeled(df)
    # TODO: Add outliers.
    # TODO: Random stuff.
    return df_result



@app.route(f'{API_ENDPOINTS["UPDATE_PROJECT_INFO"]}/<project_name>', methods=['POST'])
@cross_origin()
def update_project_info(project_name):
    """
    Update information of an existing project. This api expects json data as follows:
    {
        'project': Project name, str,
        'createDate': Project creation datetime, str,
        'description': Project description, str,
        'label': List of defined labels, List[str],
        'label_shortcut': List of defined labels, List[str],
    }

    Args:
        project_name (str): Project name.
    """
    new_info = request.get_json()
    new_description = new_info['description']
    new_label = ':sep:'.join(new_info['label'])
    new_label_shortcuts = ':sep:'.join(new_info['label_shortcut'])
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'projects.csv'))
    id = df.project == project_name
    df.loc[id, 'description'] = new_description
    df.loc[id, 'label'] = new_label
    df.loc[id, 'label_shortcut'] = new_label_shortcuts
    df.to_csv(os.path.join(PROJECT_DIR, 'projects.csv'), index=False)
    return {'success': True}, 200, {'ContentType': 'application/json'}


if __name__ == '__main__':
    app.run(debug=True)
