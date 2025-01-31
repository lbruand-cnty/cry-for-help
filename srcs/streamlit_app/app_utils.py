import os
import json
import base64
import requests
import pandas as pd
import streamlit as st
from typing import List
from datetime import datetime

from srcs import utils


def add_texts(df: pd.DataFrame, add_data: bool, text_columns: List[str],
              url: str = None):
    """
    Send a put request to add text data to a project.

    Args:
        df (pd.DataFrame): Loaded csv.
        add_data (bool): New data will be added if True (clicked "Import" button).
        text_columns (list[str]): Name of the column containing text data.
        url (str, optional): API address.
    """
    headers = {
        'content-type': 'application/json',
        'Accept-Charset': 'UTF-8',
    }
    if url is None:
        url = os.environ['API_ADDRESS'] + os.environ['ADD_DATA']

    url = f'{url}/{st.session_state.current_project}'
    if add_data and df is not None and text_columns is not None:
        extract_texts = df[text_columns].apply(lambda x: x.to_json(), axis=1).to_list()
        new_data = {'texts': extract_texts}
        r = requests.put(url, data=json.dumps(new_data), headers=headers)
        # update progress in session state if it is None
        if st.session_state.project_info['progress'] is None:
            st.session_state.project_info['progress'] = '0'


def create_project(project_name: str, url: str = None):
    """
    Send a put request to create a new project.

    Args:
        project_name (str): Project name.
        url (str, optional): API address.
    """
    if url is None:
        url = os.environ['API_ADDRESS'] + os.environ['CREATE_PROJECT']

    url = f'{url}/{project_name}'
    r = requests.put(url)


def delete_project(project_name: str, url: str = None):
    """
    Send a delete request to delete an existing project.

    Args:
        project_name (str): Project name.
        url (str, optional): API address.
    """
    if url is None:
        url = os.environ['API_ADDRESS'] + os.environ['DELETE_PROJECT']

    url = f'{url}/{project_name}'
    r = requests.delete(url)


def download_csv(project_name: str, all_or_labeled: str, url: str = None):
    """
    Send a get request to download csv of all data or just labeled data.

    Args:
        project_name (str): Project name.
        all_or_labeled (str): Set "labeled" to download labeled data or "all"
                              to download all data.
        url (str, optional): API address.
    """
    if url is None:
        url = os.environ['API_ADDRESS'] + os.environ['DOWNLOAD_DATA']

    url = f'{url}/{project_name}/{all_or_labeled}'
    r = requests.get(url)
    df = pd.DataFrame(r.json())
    csv = df.to_csv(index=False)  # if no filename is given, a string is returned
    csv = base64.b64encode(csv.encode()).decode()  # convert the csv into base64
    return csv


def get_data(url: str = None):
    """
    Send a get request to get data of the current page index and project.

    Args:
        url (str, optional): API address.
    """
    if url is None:
        url = os.environ['API_ADDRESS'] + os.environ['GET_DATA']

    url = f'{url}/{st.session_state.current_project}/{st.session_state.current_page}'
    r = requests.get(url)
    return r.json()


def get_project_info(url: str = None):
    """
    Send a get request to fetch information of current project.

    Args:
        url (str, optional): API address.
    """
    if url is None:
        url = os.environ['API_ADDRESS'] + os.environ['GET_PROJECT_INFO']

    url = f'{url}/{st.session_state.current_project}'
    r = requests.get(url)
    st.session_state.project_info = r.json()


@st.cache(show_spinner=False)
def load_config(config: str):
    """
    Load project configurations from a .yaml file.

    Args:
        config (str): Path to the configuration file.
    """
    config = utils.load_yaml(config)
    os.environ['PROJECT_DIR'] = config['PROJECT_DIR']
    os.environ['API_ADDRESS'] = config['API_ADDRESS']
    for name, value in config['API_ENDPOINTS'].items():
        os.environ[name] = value


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_projects(url: str = None) -> List[str]:
    """
    Send a get request to load list of available projects.

    Args:
        url (str, optional): API address.
    """
    if url is None:
        url = os.environ['API_ADDRESS'] + os.environ['LOAD_PROJECTS']

    r = requests.get(url)
    return r.json()['projects']


def update_label_data(new_labels: List[str], url: str = None) -> bool:
    """
    Send a put request to update the labels of the labeled data.

    Args:
        new_labels (List[str]): List of selected labels.
        queue (str): the queue in which to put the example.
        url (str, optional): API address.
    """
    headers = {
        'content-type': 'application/json',
        'Accept-Charset': 'UTF-8',
    }
    if url is None:
        url = os.environ['API_ADDRESS'] + os.environ['UPDATE_LABEL_DATA']

    url = f'{url}/{st.session_state.current_project}/{st.session_state.current_page}'
    verified = str(datetime.now()).split('.')[0][:-3] if len(new_labels) > 0 else '0'
    progress = int(st.session_state.project_info["progress"])
    if progress < 10:
        queue: str = 'test'
    elif progress < 30:
        queue: str = 'train'
    else:
        queue: str = 'train'

    data = {'new_labels': new_labels, 'verified': verified, 'queue': queue}
    # add new labels to unlabeled data
    if st.session_state.data['verified'] == '0':
        new_progress = f'{progress + 1}'
    # remove all labels from labeled data
    elif len(new_labels) == 0:
        new_progress = f'{progress - 1}'
    # change labels of labeled data
    else:
        new_progress = st.session_state.project_info['progress']

    r = requests.put(url, data=json.dumps(data), headers=headers)


    reset_page = sample_data()

    # update label and progress status into session state

    st.session_state.data = get_data()
    st.session_state.project_info['progress'] = new_progress
    return reset_page

def sample_data(url: str =None) -> bool:
    headers = {
        'content-type': 'application/json',
        'Accept-Charset': 'UTF-8',
    }
    if url is None:
        url = os.environ['API_ADDRESS'] + os.environ['SAMPLE_DATA']

    url = f'{url}/{st.session_state.current_project}'
    r = requests.post(url, data=json.dumps(st.session_state.project_info),
                      headers=headers)
    result = r.json()
    return result["reset_page"]


def update_project_info(url: str = None):
    """
    Send a post request to update project description and labels.

    Args:
        url (str, optional): API address.
    """
    headers = {
        'content-type': 'application/json',
        'Accept-Charset': 'UTF-8',
    }
    if url is None:
        url = os.environ['API_ADDRESS'] + os.environ['UPDATE_PROJECT_INFO']

    url = f'{url}/{st.session_state.current_project}'
    r = requests.post(url, data=json.dumps(st.session_state.project_info),
                      headers=headers)


def rerun():
    """ A hack to rerun streamlit app. """
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))
