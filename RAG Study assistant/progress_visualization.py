import streamlit as st
import plotly.express as px

def visualize_progress(agent):
    if not agent.progress:
        st.write("No progress data available yet.")
        return
    
    # Prepare data for bar chart
    topics = list(agent.progress.keys())
    scores = list(agent.progress.values())
    
    # Create a bar chart with Plotly
    fig = px.bar(
        x=topics,
        y=scores,
        labels={"x": "Topics", "y": "Progress Score (0-1)"},
        title="Study Progress by Topic",
        color=scores,
        color_continuous_scale="Blues",
        range_y=[0, 1],
        text=[f"{score:.2f}" for score in scores]  # Add score labels on bars
    )
    fig.update_traces(textposition='auto')  # Position labels automatically
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
