import streamlit as st

from srcs.streamlit_app import app_utils, templates, widgets

CONFIG = './config.yaml'
st.set_page_config(page_title='Cry for Help', layout='wide')


def main():
    set_session_state()
    # load project config
    app_utils.load_config(CONFIG)
    # load list of available projects
    st.session_state.projects = app_utils.load_projects()
    with st.sidebar:
        st.sidebar.title('Projects')
        project_holder = st.empty()  # placeholder to show available projects
        widgets.add_project()
        widgets.delete_project()
        # display list of available projects
        if len(st.session_state.projects) == 0:
            project_holder.write('No available project. Please add a new project.')
        else:
            try:
                i = st.session_state.projects.index(st.session_state.current_project)
            except ValueError:
                i = 0

            # ignore index if switching project (change radio button)
            if st.session_state.get('switching_project', False):
                current_project = project_holder.radio(
                    'Select a project to work with:', st.session_state.projects,
                    on_change=lambda: st.session_state.update({'switching_project': True})
                )  # set switching project state to apply the changes
                st.session_state.pop('switching_project')
            else:
                current_project = project_holder.radio(
                    'Select a project to work with:', st.session_state.projects, i,
                    on_change=lambda: st.session_state.update({'switching_project': True})
                )  # set index to persist with the selected project during pagination
            st.session_state.current_project = current_project

    left_column, _, right_column = st.columns([50, 2, 20])
    # display and update project info at the right column
    if st.session_state.current_project is not None:
        # get project info for the first time or when switching projects
        if st.session_state.project_info is None or \
                st.session_state.project_info['project'] != st.session_state.current_project:
            app_utils.get_project_info()

        with right_column:
            # display project name
            st.header(st.session_state.current_project)
            # display project creation datetime
            st.write(templates.create_date_html(st.session_state.project_info['createDate']),
                     unsafe_allow_html=True)
            # project description text area
            widgets.project_description()
            # project view_template text area
            widgets.project_view_template()
            # placeholder to display the labelling progress
            progress_holder = st.empty()
            # placeholder to display list of labels
            label_list_holder = st.empty()
            # expander to add label
            widgets.add_label()
            # expander to delete label
            widgets.delete_label()

            # import data
            file, add_data, text_columns = widgets.import_data()
            app_utils.add_texts(file, add_data, text_columns)
            # export data
            download_placeholder = widgets.export_data()

    # display data and labelling at the left column
    with left_column:
        st.title('Text Classification Data')
        if st.session_state.current_project is not None:

            keypressed = st.keypress()
            data = app_utils.get_data()

            if keypressed == 'a' and st.session_state.current_page >= 1:
                st.session_state.current_page -= 1
                data = app_utils.get_data() # If we have moved, need to reload data.
            elif keypressed == 'z' and st.session_state.current_page < data['total'] - 1:
                st.session_state.current_page += 1
                data = app_utils.get_data() # If we have moved, need to reload data.

            st.session_state.data = data
            current_page = st.session_state.current_page

            if data['total'] > 0:
                st.write(templates.page_number_html(st.session_state.current_project, current_page, data['total']),
                         unsafe_allow_html=True)
                st.write(templates.text_data_html(text=data['text'], template=st.session_state.project_info["view_template"]), unsafe_allow_html=True)
                # display checkboxes for labeling
                if len(st.session_state.project_info['label']) > 0:
                    widgets.label_data(keypressed)
                else:
                    st.write(templates.no_label_html(), unsafe_allow_html=True)
                # display the verification datetime
                if st.session_state.data['verified'] != '0':
                    st.write(templates.verified_datetime_html(st.session_state.data['verified']),
                             unsafe_allow_html=True)
            else:
                st.write('No data in this project. Please import data to start labeling.')
        else:
            st.write('No project. No data. No cry.')

    # update placeholders
    if st.session_state.current_project is not None:
        # display list of defined labels
        label_list_holder.write(
            templates.label_list_html(st.session_state.project_info['label']), # TODO: Added label_shortcuts here.
            unsafe_allow_html=True,
        )
        # display progress bar if there is data (not None)

        if st.session_state.project_info['progress'] is not None:

            total = st.session_state.data["total"]
            if total > 0:
                counts = { k: f'{ v / total * 100:.2f}' for k, v  in st.session_state.data.items() if k.startswith("total_")}
            else:
                counts = {k: f'{0:.2f}' for k, v in st.session_state.data.items() if
                          k.startswith("total_")}
            progress = int(st.session_state.project_info["progress"])
            progress = f'{progress / total * 100:.2f}' if total != 0 else '0.0'
            progress_holder.write(
                templates.progress_bar_html(counts=counts), unsafe_allow_html=True,
            )
        # display a download button upon clicking the export button
        if st.session_state.download is not None:
            download_placeholder.write(
                templates.save_csv_html(*st.session_state.download),
                unsafe_allow_html=True,
            )


def set_session_state():
    """ """
    para = st.experimental_get_query_params()
    # clicked next or previous page
    if 'page' in para.keys():
        st.experimental_set_query_params()
        new_page = max(1, int(para['page'][0])) - 1  # make sure the min is 0
        st.session_state.current_page = new_page
    if 'project' in para.keys():
        st.experimental_set_query_params()
        st.session_state.current_project = para['project'][0]

    # default values
    if 'projects' not in st.session_state:
        st.session_state.projects = []
    if 'project_info' not in st.session_state:
        st.session_state.project_info = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'download' not in st.session_state:
        st.session_state.download = None
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0


if __name__ == '__main__':
    main()
