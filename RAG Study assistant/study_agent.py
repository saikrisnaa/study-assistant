import streamlit as st

class StudyAgent:
    def __init__(self):
        # Use session state to persist progress across reruns
        if 'progress' not in st.session_state:
            st.session_state.progress = {}
        self.progress = st.session_state.progress

    def update_progress(self, topic, score):
        # Update or add the topic and score to the session state progress
        self.progress[topic] = score
        st.session_state.progress = self.progress  # Ensure session state updates

    def suggest_study_plan(self):
        suggestions = []
        for topic, score in self.progress.items():
            if score < 0.7:
                suggestions.append(f"Review '{topic}' (Score: {score:.2f})")
        return suggestions if suggestions else ["You're on track! Review any topic for practice."]
