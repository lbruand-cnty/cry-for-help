import os
import shutil
import pandas as pd
from datetime import datetime
from flask import Flask, request
from flask_cors import cross_origin
from flask_restful import Api
from typing import Dict

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

        count = { f"total_{k}":v for k, v in df.fillna("unlabeled").groupby('queue')['queue'].count().items() }

        for queue in ["unlabeled", "train", "test"]:
            if not f"total_{queue}" in count:
                count[f"total_{queue}"] = 0
        total = len(df)
        print(total)
        current_page = min(total - 1, current_page)
        text = df.texts.iloc[current_page]
        verified = str(df.verified.iloc[current_page])
        queue = str(df.queue.iloc[current_page])
        label = str(df.label.iloc[current_page])
        label = [] if label == 'nan' else label.split(':sep:')
    except FileNotFoundError:
        total = 0
        count = {}
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
        **count
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
    return inner_get_project_info(project_name)


def inner_get_project_info(project_name):
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
        'view_template': selected_df.view_template[0],
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
        'view_template': """{% for item in texts %}
    {{ item }}
{%- endfor %}""",
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
    project_info = inner_get_project_info(project_name)
    labels = project_info["label"]
    print(f"labels = {labels}")
    df = pd.read_csv(os.path.join(PROJECT_DIR, project_name, 'data.csv'))
    df_train = df[df.queue == "train"] # TODO : Queue names are hardcoded. Maybe we should have config.
    df_test = df[df.queue == "test"]
    df_unlabeled = df[ ~df.queue.isin(["train", "test"]) ]
    print(f" len(df_train) = {len(df_train)}")
    print(f" len(df_test) = {len(df_test)}")
    print(f" len(df_unlabeled) = {len(df_unlabeled)}")

    sample, rest = sample_unlabeled_data(df_unlabeled, df_train, df_test, labels)
    reset_page = (len(sample) != 0)
    df = pd.concat(
        [
            sample,
            rest,
            # TODO : This needs to be done using the model + sampling + oultliers.
            df_train,
            df_test
        ])
    df.to_csv(os.path.join(PROJECT_DIR, project_name, 'data.csv'), index=False)
    return {'success': True, 'reset_page': reset_page}, 200, {'ContentType': 'application/json'}


def sample_unlabeled_data(df_unlabeled: pd.DataFrame, df_train: pd.DataFrame, df_test: pd.DataFrame, label_list: list[str]) -> (pd.DataFrame, pd.DataFrame): # TODO : move this ml_api.
    global mlapi
    print(" == sample_unlabeled_data ==")
    if mlapi is None:
        mlapi = ml_api.MLApi()
        # TODO : There is a need to change this when we change the list of labels in the project.
        label_index: Dict[str, int] = dict(list([(v, k) for k, v in enumerate(label_list)]))
        mlapi = ml_api.MLApi(labels_index=label_index,
                             num_labels=len(label_index),
                             )
        mlapi.create_features(df_input=pd.concat( [df_unlabeled, df_train ] ))
    RETRAIN_EVERYN_VALUE = 5 # TODO: This should go in the configuration.
    if len(df_train) % RETRAIN_EVERYN_VALUE == 0 and len(df_test) >= 10 and len(df_train) >= 30:  # TODO This should go in the configuration of the project
        mlapi.train_model(training_data=df_train,
                          test_data=df_test,)
        df = df_unlabeled.sample(frac=1).reset_index(drop=True)
        # TODO: Add outliers.
        # TODO: Random stuff.
        # TODO: We should tell calling layer that every thing was change so we can move back to page 1.
        return mlapi.get_low_conf_unlabeled(df)
    else:
        return (pd.DataFrame(), df_unlabeled)



@app.route(f'{API_ENDPOINTS["UPDATE_PROJECT_INFO"]}/<project_name>', methods=['POST'])
@cross_origin()
def update_project_info(project_name):
    """
    Update information of an existing project. This api expects json data as follows:
    {
        'project': Project name, str,
        'createDate': Project creation datetime, str,
        'description': Project description, str,
        'view_template': Project template, str,
        'label': List of defined labels, List[str],
        'label_shortcut': List of defined labels, List[str],
    }

    Args:
        project_name (str): Project name.
    """
    new_info = request.get_json()
    new_description = new_info['description']
    new_view_template = new_info['view_template']
    new_label = ':sep:'.join(new_info['label'])
    new_label_shortcuts = ':sep:'.join(new_info['label_shortcut'])
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'projects.csv'))
    id = df.project == project_name
    df.loc[id, 'description'] = new_description
    df.loc[id, 'view_template'] = new_view_template
    df.loc[id, 'label'] = new_label
    df.loc[id, 'label_shortcut'] = new_label_shortcuts
    df.to_csv(os.path.join(PROJECT_DIR, 'projects.csv'), index=False)
    return {'success': True}, 200, {'ContentType': 'application/json'}


if __name__ == '__main__':
    app.run(debug=True)
