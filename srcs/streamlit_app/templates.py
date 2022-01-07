import json
from typing import List, Dict
import jinja2

def label_list_html(labels: List[str]) -> str:
    """ HTML scripts to display a list of labels. """
    html = """
            <div style="font-size:115%;font-weight:450;">
                Labels
            </div>
            <hr style="margin-top:0.5em;margin-bottom:0.5em;">
    """
    if len(labels) > 0:
        html += f"""
            <ul style="margin-bottom:1.1em;">
                {' '.join([f'<li> {label} </li>' for label in labels])}
            </ul>
        """
    else:
        html += """
            <div style="color:grey;font-size:90%;margin-bottom:1.1em;">
                Label has not been defined yet.
            </div>
        """
    return html


def no_label_html() -> str:
    """ HTML scripts to display current label and update label. """
    return """
    <div style="color:grey;font-size:90%;margin-top:1em">
        Please define a label to start labelling.
    </div>
    """


def create_date_html(date: str) -> str:
    """ HTML scripts to display the create date of a project. """
    return f"""
        <div style="color:grey;font-size:90%;">
            Created at {date}
        </div>
    """


def page_number_html(current_project: str, current_page: int,
                     total_page_number: int) -> str:
    """ HTML scripts to display the page number. """
    current_page = min(total_page_number - 1, current_page)
    html = '<div style="text-align:center;margin-top:0.3em;margin-bottom:0.5em;">'
    if current_page > 0:
        html += f'<a href="?project={current_project}&page={current_page}" style="display:inline;">&lt</a>'

    html += f"""
        <p style="display:inline;">
            &emsp;{current_page + 1}/{total_page_number}&emsp;
        </p>
    """
    if current_page < total_page_number - 1:
        html += f'<a href="?project={current_project}&page={current_page + 2}" style="display:inline;">&gt</a>'

    html += '</div>'
    return html


def progress_bar_html(counts: Dict[str, float]) -> str:
    """ HTML scripts to display progress of labelling in percentage. """
    sub_header_style = """
        font-size: 115%;
        font-weight: 450;
        margin-top: 0.3em;
        margin-bottom: 0.3em;
    """
    container_style = """
        width: 100%;
    """

    bufferbars = []
    colors = {
        "unlabeled": "rgb(128, 240, 240)",
        "train" : "rgb(128, 128, 240)",
        "test" : "orange"
    }
    texts = generate_labels(colors)

    bars = build_bars(bufferbars, colors, container_style, counts)

    sb = []
    for i, color in colors.items():
        sb.append(f"""<span style="font-size: 115%;
                font-weight: 450;
                margin-top: 0.3em;
                margin-bottom: 0.3em;
                color:{color}">{counts[f"total_{i}"]}%</span>""")
    percents = "<div>" + " / ".join(sb) + "</div>"
    return texts + bars + percents


def generate_labels(colors):
    sb = []
    for i, color in colors.items():
        sb.append(f"""<span style="font-size: 115%;
        font-weight: 450;
        margin-top: 0.3em;
        margin-bottom: 0.3em;
        color:{color}">{i}</span>""")
    texts = "<div>" + " / ".join(sb) + "</div>"
    return texts


def build_bars(bufferbars, colors, container_style, counts):
    for i in colors.keys():
        labeled_proportion = counts[f"total_{i}"]
        color = colors[i]
        progress_bar_style = f"""
            background-color: {color};
            height: 10px;
            width: {labeled_proportion}%;            
            display: inline-block;
        """
        bufferbars.append(f"""<div style="{progress_bar_style}"></div>""")
    return f"""<div style="{container_style}">""" + "".join(bufferbars) + "</div>"


def save_csv_html(filename: str, csv: str) -> str:
    """ HTML scripts to display button to save exported data in csv file. """
    return f"""
        <style>
            #save_csv {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
                margin-bottom: 1em;
            }}
            #save_csv:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #save_csv:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style>
        <a download="{filename}" id="save_csv" href="data:file/csv;base64,{csv}">
            Save to folder
        </a>
    """


def text_data_html(text: str, template: str) -> str:
    """ HTML scripts to display text to be labelled. """
    style = """
        border: none;
        border-radius: 5px;
        margin-bottom: 1em;
        padding: 20px;
        height: auto;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
    """
    t = jinja2.Template(template)
    json_text = json.loads(text)
    rendered_text = t.render(texts=json_text)
    return f"""
        <div style="{style}">
            {rendered_text}
        </div>
    """


def verified_datetime_html(date_time: str) -> str:
    """ HTML scripts to display the verification datetime. """
    return f"""
        <div style="color:grey;font-size:90%;">
            Verified at {date_time}
        </div>
    """
