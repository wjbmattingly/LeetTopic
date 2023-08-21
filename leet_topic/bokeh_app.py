from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, CustomJS, DataTable, TableColumn, MultiChoice, HTMLTemplateFormatter, TextAreaInput, Div, TextInput
from bokeh.plotting import figure, output_file, show

from bokeh.palettes import Category10, Cividis256, Turbo256
from bokeh.transform import linear_cmap

import pandas as pd
import numpy as np

from typing import Tuple, Optional
import bokeh
import bokeh.transform



#From Bulk Library
def get_color_mapping(
    df: pd.DataFrame,
    topic_field,
) -> Tuple[Optional[bokeh.transform.transform], pd.DataFrame]:
    """Creates a color mapping"""

    color_datatype = str(df[topic_field].dtype)
    if color_datatype == "object":
        df[topic_field] = df[topic_field].apply(
            lambda x: str(x) if not (type(x) == float and np.isnan(x)) else x
        )
        all_values = list(df[topic_field].dropna().unique())
        if len(all_values) == 2:
            all_values.extend([""])
        elif len(all_values) > len(Category10) + 2:
            raise ValueError(
                f"Too many classes defined, the limit for visualisation is {len(Category10) + 2}. "
                f"Got {len(all_values)}."
            )
        mapper = factor_cmap(
            field_name=topic_field,
            palette=Category10[len(all_values)],
            factors=all_values,
            nan_color="grey",
        )
    elif color_datatype.startswith("float") or color_datatype.startswith("int"):
        all_values = df[topic_field].dropna().values
        mapper = linear_cmap(
            field_name=topic_field,
            palette=Turbo256,
            low=all_values.min(),
            high=all_values.max(),
            nan_color="grey",
        )
    else:
        raise TypeError(
            f"We currently only support the following type for 'color' column: 'int*', 'float*', 'object'. "
            f"Got {color_datatype}."
        )
    return mapper, df


def create_html(df, document_field, topic_field, html_filename, topic_data, tf_idf, extra_fields=[], app_name=""):
    fields = ["x", "y", document_field, topic_field, "selected"]
    fields = fields+extra_fields
    output_file(html_filename)

    mapper, df = get_color_mapping(df, topic_field)
    df['selected'] = False
    categories = df[topic_field].unique()
    categories = [str(x) for x in categories]



    s1 = ColumnDataSource(df)


    columns = [
            TableColumn(field=topic_field, title=topic_field, width=10),
            TableColumn(field=document_field, title=document_field, width=500),
    ]
    for field in extra_fields:
        columns.append(TableColumn(field=field, title=field, width=100))


    p1 = figure(width=500, height=500, tools="pan,tap,wheel_zoom,lasso_select,box_zoom,box_select,reset", active_scroll="wheel_zoom", title="Select Here", x_range=(df.x.min(), df.x.max()), y_range=(df.y.min(), df.y.max()))
    circle_kwargs = {"x": "x", "y": "y",
                        "size": 3,
                        "source": s1,
                         "color": mapper
                        }
    scatter = p1.circle(**circle_kwargs)

    s2 = ColumnDataSource(data=dict(x=[], y=[], leet_labels=[]))
    p2 = figure(width=500, height=500, tools="pan,tap,lasso_select,wheel_zoom,box_zoom,box_select,reset", active_scroll="wheel_zoom", title="Analyze Selection", x_range=(df.x.min(), df.x.max()), y_range=(df.y.min(), df.y.max()))

    circle_kwargs2 = {"x": "x", "y": "y",
                        "size": 3,
                        "source": s2,
                         "color": mapper
                        }
    scatter2 = p2.circle(**circle_kwargs2)

    multi_choice = MultiChoice(value=[], options=categories, width = 500, title='Selection:')
    data_table = DataTable(source=s2,
                           columns=columns,
                           width=700,
                           height=500,
                          sortable=True,
                          autosize_mode='none')
    selected_texts = TextAreaInput(value = "", title = "Selected texts", width = 700, height=500)
    top_search_results = TextAreaInput(value = "", title = "Search Results", width = 250, height=500)
    top_search = TextInput(title="Topic Search")
    doc_search_results = TextAreaInput(value = "", title = "Search Results", width = 250, height=500)
    doc_search = TextInput(title="Document Search")
    topic_desc = TextAreaInput(value = "", title = "Topic Descriptions", width = 500, height=500)
    
    def field_string(field):
        return """d2['"""+field+"""'] = []\n"""

    def push_string(field):
        return """d2['"""+field+"""'].push(d1['"""+field+"""'][inds[i]])\n"""

    def indices_string(field):
        return """d2['"""+field+"""'].push(d1['"""+field+"""'][s1.selected.indices[i]])\n"""

    def push_string2(field):
        return """d2['"""+field+"""'].push(d1['"""+field+"""'][i])\n"""

    def list_creator(fields, str_type=""):
        main_str = ""
        for field in fields:
            if str_type == "field":
                main_str=main_str+field_string(field)
            elif str_type == "push":
                main_str=main_str+push_string(field)
            elif str_type == "indices":
                main_str=main_str+indices_string(field)
            elif str_type == "push2":
                main_str=main_str+push_string2(field)
        return main_str

    s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2, s4=multi_choice), code="""
            const inds = cb_obj.indices;
            const d1 = s1.data;
            const d2 = s2.data;
            const d4 = s4;"""+list_creator(fields=fields, str_type="field")+
            """for (let i = 0; i < inds.length; i++) {"""+
            list_creator(fields=fields, str_type="push")+
            """}
            const res = [...new Set(d2['"""+topic_field+"""'])];
            d4.value = res.map(function(e){return e.toString()});
            s1.change.emit();
            s2.change.emit();
        """)
    )


    multi_choice.js_on_change('value', CustomJS(args=dict(s1=s1, s2=s2, scatter=scatter, topic_desc=topic_desc, topic_data=topic_data, tf_idf=tf_idf), code="""
            let values = cb_obj.value;
            let unchange_values = cb_obj.value;
            const d1 = s1.data;
            const d2 = s2.data;
            const plot = scatter;
            s2.selected.indices = [];
            for (let i = 0; i < s1.selected.indices.length; i++) {
                for (let j =0; j < values.length; j++) {
                    if (d1."""+topic_field+"""[s1.selected.indices[i]] == values[j]) {
                        values = values.filter(item => item !== values[j]);
                    }
                }
            }
            """+list_creator(fields=fields, str_type="field")+
            """
            for (let i = 0; i < s1.selected.indices.length; i++) {
                if (unchange_values.includes(String(d1."""+topic_field+"""[s1.selected.indices[i]]))) {
                    """+
                    list_creator(fields=fields, str_type="indices")+
                    """
                }
            }
            for (let i = 0; i < d1."""+topic_field+""".length; i++) {
                if (values.includes(String(d1."""+topic_field+"""[i]))) {
                        """+
                        list_creator(fields=fields, str_type="push2")+
                        """
                }
            }
            if (tf_idf) {
                let data = [];
                for (const key of Object.keys(topic_data)) {
                    for (let i=0; i < unchange_values.length; i++) {
                        if (key == unchange_values[i]) {
                            let keywords = topic_data[key]["key_words"];
                            data.push("Topic " + key + ": ");
                            for (let i=0; i < keywords.length; i++) {
                                data.push(keywords[i][0] + " " + keywords[i][1]);
                            }
                            data.push("\\r\\n");
                        }
                    }
                }
                topic_desc.value = data.join("\\r\\n");
                s2.change.emit();
            }
        """)
    )


    s2.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2, s_texts=selected_texts), code="""
            const inds = cb_obj.indices;
            const d1 = s1.data;
            const d2 = s2.data;
            const texts = s_texts.value;
            s_texts.value = "";
            const data = [];
            for (let i = 0; i < inds.length; i++) {
                data.push(" (Topic: " + d2['"""+topic_field+"""'][inds[i]] + ")")
                data.push("Document: " + d2['"""+document_field+"""'][inds[i]])
                data.push("\\r\\n")
            }
            s2.change.emit();
            s_texts.value = data.join("\\r\\n")
            s_texts.change.emit();
        """)
    )
    
    top_search.js_on_change('value', CustomJS(args=dict(topic_data=topic_data, top_search_results=top_search_results, s4=multi_choice, s1=s1), code="""
        s1.selected.indices = []
        const search_term = cb_obj.value;
        let hits = [];
        let counter = 0;
        for (const key of Object.keys(topic_data)) {
            const keywords = topic_data[key]["key_words"];
            for (let i=0; i < keywords.length; i++) {
                if (keywords[i][0] == search_term) {
                    hits.push([key, i]);
                }
            }
        }
        hits.sort(function(a, b) {
            return a[1] - b[1];
        });
        
        const data = [];
        if (hits.length) {
            for (let i = 0; i < hits.length; i++) { 
                data.push('Topic ' + hits[i][0] + ' has "' + search_term + '" as number ' + hits[i][1] + ' in its keyword list.');
                data.push("\\r\\n");
            }
        } else if (search_term != "") {
            data.push('No keyword matches with any topic for "' + search_term + '".');
        }
        
        top_search_results.value = data.join("\\r\\n");
        
        let inds = [];
        for (let i=0; i < hits.length; i++) {
            inds.push(hits[i][0]);
        }
        
        const res = [...new Set(inds)];
        
        s4.value = res.map(function(e){return e.toString()});
    
    """)
    )
    
    doc_search.js_on_change('value', CustomJS(args=dict(s1=s1, s2=s2, df=df.to_dict(), doc_search_results=doc_search_results, s4=multi_choice), code="""
        s1.selected.indices = []
        const search_term = cb_obj.value;
        let hits = [];
        let counter = 0;
        let id_count = 0;
        for (let i = 0; i < s1.data.top_words.length; i++) {
            for (let j = 0; j <s1.data.top_words[i].length; j++) { 
                if (search_term == s1.data.top_words[i][j][0]) { 
                    hits.push([id_count, j]);
                }
                
            }
            id_count = id_count + 1;
        }
        
        hits.sort(function(a, b) {
            return a[1] - b[1];
        });
        
        const data = [];
        if (hits.length) {
            for (let i = 0; i < hits.length; i++) { 
                data.push('Document ' + hits[i][0] + ' has "' + search_term + '" as number ' + hits[i][1] + ' in its top_words list.');
                data.push("\\r\\n");
            }
        } else if (search_term != "") {
            data.push('No keyword matches with any document for "' + search_term + '".');
        }
        
        doc_search_results.value = data.join("\\r\\n");
        
        let inds = [];
        for (let i=0; i <hits.length; i++) {
            inds.push(hits[i][0]);
            s1.selected.indices.push(hits[i][0]);
        }
        
        
        
        const d1 = s1.data;
        const d2 = s2.data;
        const d4 = s4;"""+list_creator(fields=fields, str_type="field")+
        """for (let i = 0; i < inds.length; i++) {"""+
        list_creator(fields=fields, str_type="push")+
        """}
        const res = [...new Set(d2['"""+topic_field+"""'])];
        s4.value = res.map(function(e){return e.toString()});
        s1.change.emit();
        s2.change.emit();
        
    
    """)
    )

    if tf_idf:
        col1 = column(p1, multi_choice, topic_desc) 
    else:
        col1 = column(p1, multi_choice) 
    col2 = column(data_table, selected_texts)
    if tf_idf:
        col3 = column(p2, row(column(doc_search, doc_search_results), column(top_search, top_search_results)))
    else:
        col3 = column(p2)
    app_row = row(col1, col2, col3)
    if app_name != "":
        title = Div(text=f'<h1 style="text-align: center">{app_name}</h1>')
        layout = column(title, app_row, sizing_mode='scale_width')
    else:
        layout=app_row
    show(layout)