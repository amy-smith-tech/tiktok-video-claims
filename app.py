import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from huggingface_hub import hf_hub_download
import joblib

# from utils import Preprocessor # update the preprocessing pipeline in utils.py

REPO_ID = "amy-smith-tech/tiktok-claims-rfc"
FILENAME = "random_forest_model_amy.joblib"

# Helper function for prediction
def predict_review(input):
    
    # Predict sentiment
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = joblib.load(model_path)

    #TODO : read a pandas dataframe of inputs to the model
    #
    #
    #
    
    prediction = model.predict(input)
    prediction_prob = model.predict_proba(input)[0]
    
    return prediction, prediction_prob

def run():
    # Streamlit UI
    st.set_page_config(page_title="TikTok Claims Predictor", layout="centered")
    
    # Header
    st.title("TikTok Claims Predictor")
    st.subheader("Analyze and predict the claims of TikTok videos.")
    
    # User Input
    st.markdown("### Enter Your Claim Report")
    user_review = st.text_area(
        "Type or paste a claim report review below to predict its claim status.",
        placeholder="i think that drone deliveries are already happening and will become common by 2025",
    )

    verified_options = [0, 1]
    ban_options = [0, 1]
    
    video_duration_sec = st.sidebar.slider("Video duration sec:", 1, 100, value=46)
    verified_status = st.sidebar.radio("Verified Options", verified_options)
    ban_status = st.sidebar.radio("Ban status:", ban_options)

    # TODO: Add widgets to obtain information about the Tiktok video
    #
    #
    #

    data = {
    "video_duration_sec": video_duration_sec,
    "verified_status": verified_status,
    "author_ban_status": ban_status,
    "video_view_count": 532092,
    "video_like_count": 269442,
    "video_share_count": 77939,
    "video_download_count": 5624,
    "video_comment_count": 28}

    data = pd.DataFrame([data])
    
    # Submit Button
    if st.button("Predict claim status"):
        if user_review.strip():
            # Make prediction
            prediction, prediction_prob = predict_review(data)
            claim = "Opinion" if prediction == 1 else "Claim"
            prob_positive = round(prediction_prob[1] * 100, 2)
            prob_negative = round(prediction_prob[0] * 100, 2)
    
            # Display Results
            st.markdown(f"### Claim: **{claim}**")
            st.markdown(f"**Confidence:** {prob_positive}% Opinion, {prob_negative}% Claim")
            
            # Plotly Bar Chart for Probabilities
            fig = go.Figure(data=[
                go.Bar(
                    x=["Opinion", "Claim"],
                    y=[prob_positive, prob_negative],
                    text=[f"{prob_positive}%", f"{prob_negative}%"],
                    textposition='auto',
                    marker=dict(color=['red', 'green'])
                )
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Claim or Opinion",
                yaxis_title="Probability (%)",
                template="plotly_white"
            )
            st.plotly_chart(fig)
            
            st.info(
                "Reporting a claim is an important tool for maintaining a safe and respectful environment on social media platforms."
            )
        else:
            st.error("Please enter a valid review before clicking 'Predict'.")
    
    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit | © 2024 Hotel Insights AI")

if __name__ == "__main__":
    run()

"""
    Scrap code
"""

# from random import randint

# import altair as alt
# import pandas as pd
# import streamlit as st

# # Input widgets
# side_options = [6, 10, 12, 20]
# num_sides = st.sidebar.radio("Number of sides:", side_options)
# num_dice = st.sidebar.slider("Number of dice:", 1, 10, value=2)
# num_rolls_sim = st.sidebar.slider("Number of rolls in simulation",
#         1_000, 100_000, value=1_000, step=1_000)

# # Roll calculation
# rolls = [randint(1, num_sides) for _ in range(num_dice)]
# roll = sum(rolls)

# # Simulation rolls
# sim_rolls = []
# for _ in range(num_rolls_sim):
#     sim_roll = sum(
#         [randint(1, num_sides) for _ in range(num_dice)])
#     sim_rolls.append(sim_roll)
# df_sim = pd.DataFrame({"rolls": sim_rolls})

# # Create histogram
# chart = alt.Chart(df_sim).mark_bar().encode(
#     alt.X("rolls", bin=True),
#     y="count()",
# )
# chart.title = f"Simulation of {num_rolls_sim} rolls"

# # Main page
# st.title("Rolling Dice")
# st.button("Roll!")

# st.write("---")
# st.subheader(roll)
# st.write(str(rolls))

# st.write("---")
# st.altair_chart(chart)
