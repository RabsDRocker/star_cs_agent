# Import the necessary libraries
import streamlit as st # The main library for creating the web app
import json           # To read and parse the JSON report file
import os             # To check if the report file exists
import time           # To pause between updates

# --- Configuration ---
# Define the path to the report file. This must match the path in your reasoning_api_server.py
REPORT_FILE = os.path.join(os.path.dirname(__file__), '../data/safety_report.json')

# --- Streamlit App Configuration ---
# Set the page layout to "wide" to use the full screen width and give it a title
st.set_page_config(layout="wide", page_title="STAR-CS Live Dashboard")

# --- Main App Interface ---

# Display the main title of the dashboard
st.title("STAR-CS: Spatio-Temporal Agentic Reasoning for Construction Safety")
st.write("This dashboard displays real-time safety alerts and recommendations from the AI agent.")

# Create a placeholder. This is a key Streamlit feature.
# It creates an empty container that we can overwrite repeatedly,
# which makes the dashboard feel like it's "updating" in place instead of
# just printing new reports over and over.
placeholder = st.empty()

# --- Main Loop ---
# This loop runs forever, constantly checking for new reports.
while True:
    # Check if the report file has been created by the AI agent
    if os.path.exists(REPORT_FILE):
        # If the file exists, open and read it
        with open(REPORT_FILE, "r") as f:
            try:
                report = json.load(f)
            except json.JSONDecodeError:
                # Handle cases where the file might be being written at the exact same time
                report = {}
                continue
        
        # Use the placeholder to draw the dashboard content
        with placeholder.container():
            # Display a prominent alert header with the event type and timestamp
            st.error(f"**ALERT: {report.get('event_type', 'N/A')}** at {time.ctime(report.get('timestamp', 0))}")
            
            # Show the detailed, human-readable description of the event
            st.info(f"**Details:** {report.get('details', 'No details provided.')}")
            
            # Create two columns for a cleaner layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Display the identified violations in the left column
                st.subheader("Identified Violations")
                # st.markdown() is used to render formatted text
                st.markdown(report.get('identified_violations', 'None'))
            
            with col2:
                # Display the actionable recommendations in the right column
                st.subheader("Actionable Recommendations")
                st.warning(report.get('actionable_recommendations', 'None'))
    else:
        # If the report file does not exist, display a waiting message
        with placeholder.container():
            st.info("System is running... Waiting for a safety event from the construction site.")
            
    # Pause for 2 seconds before checking for the file again.
    # This prevents the script from using 100% of the CPU.
    time.sleep(2)