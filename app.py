import streamlit as st
import pandas as pd
from model_utils import load_artifacts, predict

st.set_page_config(page_title="POS & NER Tagger", page_icon="🏷️", layout="centered")

@st.cache_resource
def get_model():
    return load_artifacts()

# ── Load with error feedback ─────────────────────────────────────────
try:
    model, tok_sen, tok_pos, tok_ner, max_len = get_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Model failed to load: {e}")
    model_loaded = False

# ── UI ───────────────────────────────────────────────────────────────
st.title("🏷️ POS & NER Tagger")
st.markdown("Enter a sentence to predict **Part-of-Speech** and **Named Entity** tags.")

sentence = st.text_area("Input sentence", placeholder="e.g. Barack Obama was born in Hawaii.", height=100)

if st.button("Predict", type="primary", disabled=not model_loaded):
    if not sentence.strip():
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Running model..."):
            try:
                results = predict(sentence, model, tok_sen, tok_pos, tok_ner, max_len)

                df = pd.DataFrame(results)

                st.subheader("Results Table")
                st.dataframe(df, use_container_width=True)

                # ── Colour-coded NER highlight ───────────────────────
                st.subheader("Entity Highlight")
                NER_COLORS = {
                    "B-PER": "#FFDDC1", "I-PER": "#FFDDC1",
                    "B-ORG": "#C1E1FF", "I-ORG": "#C1E1FF",
                    "B-LOC": "#C1FFC1", "I-LOC": "#C1FFC1",
                    "O":     "#F0F0F0",
                }

                html = ""
                for row in results:
                    color = NER_COLORS.get(row["NER"], "#E8E8E8")
                    pos_tag = row["POS"]
                    ner_tag = row["NER"]
                    token   = row["Token"]
                    html += (
                        '<span style="background:' + color + ';border-radius:4px;'
                        'padding:3px 8px;margin:3px;display:inline-block;'
                        'font-weight:600;font-size:15px" '
                        'title="POS: ' + pos_tag + ' | NER: ' + ner_tag + '">'
                        + token + '</span>'
                    )

                st.markdown(html, unsafe_allow_html=True)
                st.caption("💡 Hover over tokens to see POS + NER tags.")

            except Exception as e:
                st.error(f"Prediction error: {e}")