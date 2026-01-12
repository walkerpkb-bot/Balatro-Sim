---
name: streamlit-ui
description: Build and improve Streamlit interfaces with polished UX, responsive layouts, and visual appeal
---

# Streamlit UI Skill

Build beautiful, functional Streamlit apps that don't look like default demos.

## Layout Principles

### Use columns for side-by-side content
```python
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Score", "1,234")
with col2:
    st.metric("Win Rate", "67%")
```

### Use expanders for detail views
```python
with st.expander("View Details"):
    st.write("Hidden content here")
```

### Use containers for grouped content
```python
with st.container():
    st.subheader("Section")
    # grouped elements
```

### Use sidebar for controls, main area for results
```python
with st.sidebar:
    option = st.selectbox("Choose", options)
# Main area shows results
```

## Visual Polish

### Custom CSS for styling
```python
st.markdown("""
<style>
    .stMetric { background: #1e1e1e; padding: 10px; border-radius: 5px; }
    .stExpander { border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)
```

### Use st.metric for KPIs (not plain text)
```python
st.metric("Revenue", "$12,345", delta="+5%")
```

### Display images responsively
```python
st.image(url, width=120)  # Control size
cols = st.columns(4)
for i, img in enumerate(images):
    with cols[i % 4]:
        st.image(img, width=80)
```

### Progress and status indicators
```python
with st.spinner("Loading..."):
    result = expensive_operation()
st.success("Done!")
st.error("Failed")
st.warning("Check this")
st.info("Note")
```

## Performance

### Cache data loading
```python
@st.cache_data
def load_data():
    return pd.read_csv("large_file.csv")

@st.cache_resource
def get_model():
    return load_expensive_model()
```

### Use session state for persistence
```python
if "counter" not in st.session_state:
    st.session_state.counter = 0
st.session_state.counter += 1
```

## Interactive Elements

### Buttons with feedback
```python
if st.button("Run", type="primary", use_container_width=True):
    with st.spinner("Running..."):
        result = run_simulation()
    st.success("Complete!")
```

### Forms for grouped inputs
```python
with st.form("my_form"):
    name = st.text_input("Name")
    submitted = st.form_submit_button("Submit")
    if submitted:
        process(name)
```

## Data Display

### DataFrames with config
```python
st.dataframe(df, use_container_width=True, hide_index=True)
```

### Charts
```python
st.bar_chart(data)
st.line_chart(data)
# Or use plotly for more control
import plotly.express as px
fig = px.bar(df, x="category", y="value")
st.plotly_chart(fig, use_container_width=True)
```

## Common Patterns

### Two-panel layout (controls + results)
```python
left, right = st.columns([1, 2])
with left:
    st.subheader("Settings")
    # controls
with right:
    st.subheader("Results")
    # output
```

### Card-like display
```python
with st.container():
    cols = st.columns([1, 4])
    with cols[0]:
        st.image(icon_url, width=50)
    with cols[1]:
        st.markdown(f"**{title}**")
        st.caption(description)
```

### Tabbed interface
```python
tab1, tab2 = st.tabs(["Overview", "Details"])
with tab1:
    st.write("Overview content")
with tab2:
    st.write("Detailed content")
```

## Avoid

- Don't use st.write() for everything - use specific widgets
- Don't put too many elements without visual grouping
- Don't forget to set page_config at the top
- Don't use default column widths when unequal makes more sense
- Don't show raw data without formatting (use st.metric, st.dataframe)
