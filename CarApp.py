"""CarApp — 2025 Great Britain road collision analysis, labelled dataset.

Works directly with the descriptive values in the user's Department for
Transport 2025 collision CSV. Run: streamlit run CarApp.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.set_page_config(page_title='CarApp | 2025 Road Collisions', page_icon='🚗', layout='wide')
ROOT = Path(__file__).parent
DATA = Path('/content/road_collision_uploaded.csv')
FEATURES = ['speed_limit', 'road_type', 'light_conditions',
            'weather_conditions', 'road_surface_conditions', 'urban_or_rural_area']
CATEGORICAL = [f for f in FEATURES if f != 'speed_limit']
PLOTS = {
    'Weather conditions': 'weather_conditions',
    'Lighting conditions': 'light_conditions',
    'Road surface': 'road_surface_conditions',
    'Road type': 'road_type',
    'Urban or rural': 'urban_or_rural_area',
    'Speed limit': 'speed_limit',
}
SEVERITIES = ['Slight', 'Serious', 'Fatal']

st.markdown('''
<style>
.block-container {padding-top: 1.3rem; max-width: 1280px;}
/* Make the metric values readable in both light and dark browser themes. */
[data-testid="stMetric"] {
  background: #f3f7fa !important;
  border: 1px solid #dae5ed;
  padding: 16px;
  border-radius: 12px;
}
[data-testid="stMetric"],
[data-testid="stMetric"] * {
  color: #173047 !important;
  -webkit-text-fill-color: #173047 !important;
}
.hero {background: linear-gradient(120deg, #142c48, #147580); border-radius: 16px;
  padding: 26px 30px; color: #fff; margin-bottom: 20px;}
.hero h1 {color: #fff; margin: 0; font-size: 34px;}
.hero p {color: #e2f0f1; margin: 8px 0 0 0;}
</style>
<div class="hero"><h1>🚗 CarApp — Road Collision Analysis</h1>
<p>Great Britain, 2025 · Data exploration and Decision Tree classification</p></div>
''', unsafe_allow_html=True)

@st.cache_data(show_spinner='Loading the 2025 collision dataset…')
def load_data():
    return pd.read_csv(DATA, low_memory=False)

@st.cache_resource(show_spinner='Training Decision Tree using labelled categories…')
def train_tree():
    # Same settings as the coursework notebook. Everything is fitted using
    # the training split, not the held-out test records.
    df = load_data()
    required = FEATURES + ['collision_severity']
    missing = [col for col in required if col not in df]
    if missing:
        raise ValueError(f'Missing dataset columns: {missing}')
    valid = df['collision_severity'].isin(SEVERITIES)
    X = df.loc[valid, FEATURES].copy()
    X['speed_limit'] = pd.to_numeric(X['speed_limit'], errors='raise')
    y = df.loc[valid, 'collision_severity'].isin(['Serious', 'Fatal']).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer([
        ('categories', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('speed', 'passthrough', ['speed_limit']),
    ])
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=100,
            class_weight='balanced', random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, y_test, pred, len(X_train), len(X_test)

if not DATA.exists():
    st.warning('Please upload your labelled CSV in the sidebar, or run the first Colab cell.')
    uploaded_csv = st.sidebar.file_uploader('Choose labelled CSV', type=['csv'])
    if uploaded_csv is None:
        st.stop()
    DATA.write_bytes(uploaded_csv.getvalue())
    st.rerun()

try:
    df = load_data()
except Exception as error:
    st.error(f'Could not read the included dataset: {error}')
    st.stop()

st.sidebar.header('Explore the records')
st.sidebar.success('✅ Labelled CSV loaded.')
st.sidebar.caption(f'{len(df):,} police-reported injury collision records (2025).')
st.sidebar.divider()
st.sidebar.markdown('**Source:** Department for Transport, 2025 road safety open data and its supplied data guide.')

levels = df['collision_severity'].value_counts()
metric_cols = st.columns(4)
for col, title, value in zip(metric_cols,
                             ['Total collisions', 'Slight', 'Serious', 'Fatal'],
                             [len(df), int(levels.get('Slight', 0)),
                              int(levels.get('Serious', 0)), int(levels.get('Fatal', 0))]):
    col.metric(title, f'{value:,}')

summary, exploration, quality, modelling = st.tabs([
    '📊 Overview', '🔎 Collision Analysis', '🧹 Data Preprocessing', '🌳 Decision Tree'
])

with summary:
    a, b = st.columns(2)
    with a:
        st.subheader('Collision severity')
        counts = levels.reindex(SEVERITIES, fill_value=0).reset_index()
        counts.columns = ['Severity', 'Collisions']
        fig = px.bar(counts, x='Severity', y='Collisions', color='Severity',
                     color_discrete_map={'Slight': '#318C82', 'Serious': '#E6A14A', 'Fatal': '#C9535D'},
                     text='Collisions')
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader('Speed limits')
        counts = df['speed_limit'].value_counts().sort_index()
        fig = px.bar(x=[f'{x} mph' for x in counts.index], y=counts.values,
                     labels={'x': 'Speed limit', 'y': 'Collisions'})
        fig.update_traces(marker_color='#287B98')
        fig.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader('Dataset preview — descriptive categories')
    show_cols = ['collision_index', 'date', 'collision_severity', 'weather_conditions',
                 'light_conditions', 'road_surface_conditions', 'road_type', 'speed_limit']
    st.dataframe(df[[c for c in show_cols if c in df]].head(20),
                 use_container_width=True, hide_index=True)
    st.caption('Speed limits and collision counts remain numerical because they are measurements.')

with exploration:
    st.subheader('How conditions relate to recorded collision severity')
    a, b, c = st.columns(3)
    with a:
        factor_name = st.selectbox('Condition to examine', list(PLOTS))
    with b:
        selected = st.selectbox('Severity', ['Serious or Fatal', 'Fatal', 'Serious', 'Slight', 'All'])
    with c:
        metric = st.selectbox('Measure', ['Number of collisions', 'Share within condition (%)'])
    col_name = PLOTS[factor_name]
    selected_mask = (
        df['collision_severity'].isin(['Serious', 'Fatal']) if selected == 'Serious or Fatal' else
        pd.Series(True, index=df.index) if selected == 'All' else
        df['collision_severity'].eq(selected)
    )
    keys = df[col_name].fillna('Not recorded').astype(str)
    if col_name == 'speed_limit':
        keys = keys + ' mph'
    totals = keys.value_counts()
    matched = keys[selected_mask].value_counts()
    data = pd.DataFrame({'Condition': totals.index,
                         'All recorded collisions': totals.values,
                         'Selected collisions': matched.reindex(totals.index, fill_value=0).values})
    data['Share within condition (%)'] = (100 * data['Selected collisions'] /
                                          data['All recorded collisions']).round(2)
    if selected == 'All':
        metric = 'Number of collisions'
    graph_col = ('Selected collisions' if metric == 'Number of collisions'
                 else 'Share within condition (%)')
    data = data.sort_values(graph_col, ascending=False)
    st.plotly_chart(px.bar(data, x='Condition', y=graph_col, color_discrete_sequence=['#217F91']),
                    use_container_width=True)
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.caption('Percentages describe severity among *recorded injury collisions* in each condition. '
               'They do not measure the likelihood of an accident, and do not establish causation.')

with quality:
    st.subheader('Data quality checks')
    a, b, c = st.columns(3)
    a.metric('Rows', f'{df.shape[0]:,}')
    b.metric('Columns', f'{df.shape[1]}')
    c.metric('Exact duplicate rows', f'{int(df.duplicated().sum()):,}')
    missing = df.isnull().sum().sort_values(ascending=False)
    st.dataframe(pd.DataFrame({'Column': missing.index, 'Missing values': missing.values,
                               'Missing %': (missing.values / len(df) * 100).round(4)}),
                 hide_index=True, use_container_width=True)
    st.info('Some columns may contain category descriptions such as “Unknown” or '
            '“Data missing or out of range” rather than empty cells. '
            'A few local-authority codes lacked a matching guide entry and were left unchanged.')

with modelling:
    st.subheader('Decision Tree — Slight vs Serious/Fatal')
    st.write('The model uses six road and environmental variables, with an 80% training '
             'and 20% testing split. Text categories are converted using One-Hot Encoding.')
    try:
        model, y_test, pred, n_train, n_test = train_tree()
        from sklearn.metrics import recall_score, f1_score
        a, b, c = st.columns(3)
        a.metric('Test accuracy', f'{accuracy_score(y_test, pred):.2%}')
        b.metric('Serious/Fatal recall', f'{recall_score(y_test, pred):.2%}')
        c.metric('Serious/Fatal F1-score', f'{f1_score(y_test, pred):.2f}')
        st.caption(f'Training records: {n_train:,} · Testing records: {n_test:,}. '
                   'Model depth: 5; minimum leaf size: 100; balanced class weights.')
        left, right = st.columns([1.0, 1.0])
        with left:
            st.markdown('**Confusion matrix**')
            # Draw on a white figure with high-contrast numbers and tick labels.
            with plt.style.context('default'):
                fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor='white')
                cm = confusion_matrix(y_test, pred)
                disp = ConfusionMatrixDisplay(
                    cm, display_labels=['Slight', 'Serious/Fatal']
                )
                disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        disp.text_[i, j].set_color(
                            '#ffffff' if cm[i, j] >= cm.max() * 0.55 else '#142c48'
                        )
                        disp.text_[i, j].set_fontweight('bold')
                ax.set_facecolor('white')
                ax.tick_params(colors='#142c48')
                ax.xaxis.label.set_color('#142c48')
                ax.yaxis.label.set_color('#142c48')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        with right:
            st.markdown('**Classification report**')
            report = pd.DataFrame(classification_report(
                y_test, pred, target_names=['Slight', 'Serious/Fatal'],
                output_dict=True, zero_division=0)).T.round(3)
            st.dataframe(report, use_container_width=True)
            st.warning('The model misses around half of Serious/Fatal collisions, so it '
                       'is useful for demonstrating classification, not for reliable safety decisions.')
        with st.expander('View simplified Decision Tree diagram'):
            with plt.style.context('default'):
                fig, ax = plt.subplots(figsize=(19, 9), facecolor='white')
                feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                feature_names = [name.replace('categories__', '').replace('speed__', '')
                                 for name in feature_names]
                plot_tree(model.named_steps['classifier'], feature_names=feature_names,
                          class_names=['Slight', 'Serious/Fatal'], max_depth=3,
                          rounded=True, filled=True, fontsize=7, ax=ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            st.caption('Shows only the first three levels; the trained tree can reach depth five.')
    except Exception as error:
        st.error(f'The Decision Tree could not be evaluated: {error}')
