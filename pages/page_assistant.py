# === FILE: pages/page_assistant.py ===
import os
import streamlit as st
import requests

def query_huggingface_free(prompt, model_id="tiiuae/falcon-7b-instruct"):
    try:
        API_URL = f"https://api-inference.huggingface.co/models/{model_id}"

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()

            if isinstance(result, list):
                return result[0].get("generated_text", "")
            return str(result)

        else:
            print("HF ERROR:", response.status_code, response.text)
            return None

    except Exception as e:
        print("HF EXCEPTION:", str(e))
        return None

def get_local_response(user_input):
    """Fallback local response system"""
    q = user_input.lower()
    
    if any(word in q for word in ['good gvi', 'gvi score', 'what is gvi', 'green visibility']):
        return """📊 **What is a good GVI score?**
        
The Green Visibility Index (GVI) measures vegetation coverage as a percentage of total area.

- **Excellent (40%+):** Dense urban forest, excellent cooling potential
- **Good (30-40%):** Healthy green coverage, good equity likely
- **Moderate (20-30%):** Room for improvement, prioritize underserved areas
- **Low (10-20%):** Urgent intervention needed
- **Very Low (<10%):** Critical green space deficit"""
    
    elif 'gini' in q:
        return """⚖️ **What does the Gini coefficient mean for green space?**
        
The Gini coefficient measures how evenly green space is distributed across a city:

- **0.0-0.3:** Excellent equity - green space is well distributed
- **0.3-0.5:** Moderate equity - some patches have less greenery
- **0.5-0.7:** Poor equity - significant inequality
- **0.7-1.0:** Severe inequality - urgent intervention needed"""
    
    elif 'cooling' in q or 'temperature' in q:
        return """🌡️ **How is cooling potential calculated?**
        
Based on Bowler et al. (2010) meta-analysis:
**Cooling = (Patch GVI / 10) × 0.3°C**

This means:
- 10% GVI → 0.3°C cooling
- 30% GVI → 0.9°C cooling  
- 50% GVI → 1.5°C cooling"""
    
    elif 'hotspot' in q or 'intervention' in q:
        return """📍 **What are greenery hotspots?**
        
Hotspots are 4x4 grid patches flagged for intervention based on:
1. High Discrepancy between aerial and street GVI
2. High Uncertainty in model predictions
3. Low Equity Score compared to other patches"""
    
    elif 'uncertainty' in q or 'confidence' in q:
        return """🎯 **How does uncertainty weighting work?**
        
Our system uses Monte Carlo Dropout with 20 stochastic forward passes:
- Low Uncertainty → Trust aerial view more
- Medium Uncertainty → Balanced weights
- High Uncertainty → Trust street view more"""
    
    elif 'improve' in q or 'increase' in q or 'strategy' in q:
        return """🌱 **Strategies to Improve Urban Green Space**
        
**Short-term:** Convert vacant lots, install green roofs, plant street trees
**Medium-term:** Develop green corridors, implement tree canopy targets
**Long-term:** Integrate green space into urban planning, protect existing canopy"""
    
    else:
        return """💡 **Urban Green Space Assistant**

I can help you understand:
- GVI Scores - What is a good green visibility index?
- Gini Coefficient - How equitable is green space distribution?
- Cooling Potential - How much temperature reduction is possible?
- Hotspot Detection - Where should interventions focus?
- Improvement Strategies - Actionable recommendations

Ask a specific question about any of these topics!"""

def show():
    """Main assistant page function - no arguments required"""
    
    st.markdown("## 🤖 Urban Green Space Assistant")
    st.markdown("Ask questions about GVI, equity, cooling potential, and urban greening strategies")
    
    # Initialize chat history
    if 'assistant_history' not in st.session_state:
        st.session_state.assistant_history = []
    
    # Display chat history
    for msg_type, text in st.session_state.assistant_history:
        if msg_type == 'user':
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").write(text)
    
    # Suggested questions as buttons
    st.markdown("### 💬 Suggested Questions")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    suggestions = [
        ("📊 Good GVI score?", "gvi"),
        ("⚖️ What is Gini?", "gini"),
        ("🌡️ Cooling calculation?", "cooling"),
        ("📍 What are hotspots?", "hotspots"),
        ("🌱 Improve green space?", "improve")
    ]
    
    user_input = None
    for col, (label, key) in zip([col1, col2, col3, col4, col5], suggestions):
        if col.button(label, key=key, use_container_width=True):
            user_input = label
            break
    
    if user_input is None:
        user_input = st.chat_input("Ask a question about urban green spaces...")
    
    if user_input:
        # Add user message to history
        st.session_state.assistant_history.append(('user', user_input))
        st.chat_message("user").write(user_input)
        
        # Get response
        with st.spinner("Thinking..."):
            response = query_huggingface_free(f"User question about urban green space: {user_input}")
            if response is None:
                response = get_local_response(user_input)
        
        # Add assistant response to history
        st.session_state.assistant_history.append(('assistant', response))
        st.chat_message("assistant").write(response)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.assistant_history = []
        st.rerun()

if __name__ == "__main__":
    show()