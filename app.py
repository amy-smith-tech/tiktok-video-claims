import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re

from huggingface_hub import hf_hub_download
import joblib

# Change RepoID and model filename here
REPO_ID = "amy-smith-tech/tiktok-claims-rfc"
FILENAME = "random_forest_model_v2.joblib"

opinion_keywords = ["think", "feel", "opinion", "believe"] # words that likely appear in opinions
claim_keywords = ["shared"] # words that likely appear in claims
opinion_keywords = r"|".join(opinion_keywords)
claim_keywords = r"|".join(claim_keywords)

# Helper function for prediction
def predict_review(input):
    
    # Predict on your review claims here
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = joblib.load(model_path)
    prediction = model.predict(input)
    prediction_prob = model.predict_proba(input)[0]
    
    return prediction, prediction_prob

def run():
    # Streamlit UI
    st.set_page_config(page_title="TikTok Claims Predictor", layout="centered")
    
    # Header
    st.title("TikTok Claims Predictor")
    st.subheader("Analyze and predict the claims of TikTok videos.")

    # Examples for testing
    example_claims = [
        "Drone deliveries are already happening and will become common by 2025.",
        "someone shared with me that there are more microorganisms in one teaspoon of soil than people on the planet",
        "I think that 90% of goods are shipped by ocean freight",
        "i think that 20% of the world's oxygen is produced in the amazon jungles",
        "someone shared with me that american industrialist andrew carnegie had a net worth of $475 million usd, worth over $300 billion usd today"
    ]
    
    # Example Selector
    st.markdown("### Choose or Enter a Claim Report")
    selected_example = st.selectbox("Choose an example claim (or type your own below):", [""] + example_claims)
    
    # User Input
    user_review = st.text_area(
        "Type or paste a claim report review below to predict its claim status.",
        value=selected_example,
        placeholder="i think that drone deliveries are already happening and will become common by 2025",
    )

    verified_options = [0, 1]
    ban_options = [0, 1]

    # Added customization: Add widgets to obtain information about the Tiktok video
    video_duration_sec = st.sidebar.slider("Video duration sec:", 1, 60, value=1)
    verified_status = st.sidebar.radio("Verified Options", verified_options)
    ban_status = st.sidebar.radio("Ban status:", ban_options)
    video_view_count = st.sidebar.slider("Video view counts:", 1, 1000000, value=1)
    video_like_count = st.sidebar.slider("Video like counts:", 1, 700000, value=1)
    video_share_count = st.sidebar.slider("Video share counts:", 1, 700000, value=1)
    video_download_count = st.sidebar.slider("Video download counts:", 1, 15000, value=1)
    video_comment_count = st.sidebar.slider("Video comment counts:", 1, 10000, value=1)

    # Count occurrences of keywords
    pattern = opinion_keywords
    video_opinion_kw_count = len(re.findall(pattern, user_review, flags=re.IGNORECASE))
    pattern = claim_keywords
    video_claim_kw_count = len(re.findall(pattern, user_review, flags=re.IGNORECASE))
    
    data = {
    "video_duration_sec": video_duration_sec,
    "verified_status": verified_status,
    "author_ban_status": ban_status,
    "video_view_count": video_view_count,
    "video_like_count": video_like_count,
    "video_share_count": video_share_count,
    "video_download_count": video_download_count,
    "video_comment_count": video_comment_count,
    "video_opinion_kw_count": video_opinion_kw_count,
    "video_claim_kw_count": video_claim_kw_count,
    "video_transcription_len": len(user_review)}

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
    st.markdown("Developed with ❤️ using Streamlit | © 2025 Amy Smith Tech AI")

if __name__ == "__main__":
    run()
