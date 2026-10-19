import streamlit as st
from PIL import Image
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from streamlit_mic_recorder import mic_recorder

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Medical & Agriculture Assistant",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# 2. LOAD GEMINI API KEY
# ============================================================

load_dotenv()
api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))


if not api_key:
    st.error("❌ GEMINI_API_KEY was not found in your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.5-flash-lite"


# ============================================================
# 3. GEMINI HELPER FUNCTION
# ============================================================

def ask_gemini(contents, retries=3):

    last_error = None

    for attempt in range(retries):

        try:

            result = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents
            )

            if result is not None and result.text:

                return result.text

            return (
                "⚠️ Gemini returned an empty response. "
                "Please try again."
            )

        except Exception as e:

            last_error = e
            error_text = str(e).upper()

            # ------------------------------------------------
            # API QUOTA ERROR
            # ------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                return (
                    "⚠️ Gemini API quota has been reached.\n\n"
                    "Your current Gemini API request quota is exhausted. "
                    "Please wait until the quota resets or use an API "
                    "project with available quota."
                )

            # ------------------------------------------------
            # TEMPORARY SERVER / CONNECTION ERROR
            # ------------------------------------------------

            temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "INTERNAL" in error_text
                or "TIMEOUT" in error_text
                or "CONNECTIONRESETERROR" in error_text
                or "CONNECTION RESET" in error_text
            )

            if temporary_error and attempt < retries - 1:

                wait_time = 5 * (attempt + 1)

                time.sleep(wait_time)

                continue

            break

    return (
        "⚠️ Gemini is temporarily unavailable.\n\n"
        "Please try again later.\n\n"
        f"Technical error: {last_error}"
    )


# ============================================================
# 4. TITLE
# ============================================================

st.title("🩺 AI Medical & Agriculture Assistant")

st.write(
    "AI-powered assistance for medical and agriculture-related "
    "questions and image analysis."
)

st.warning(
    "⚠️ Demo only: Medical responses are for educational purposes "
    "and are not a confirmed diagnosis. For medical concerns, "
    "consult a qualified healthcare professional."
)


# ============================================================
# 5. SPECIALIST SELECTION
# ============================================================

st.sidebar.header("🔍 Select Specialist")

specialist = st.sidebar.selectbox(
    "Choose a specialist",
    [
        "Medical - Eye",
        "Medical - Chest X-Ray",
        "Medical - Bone X-Ray",
        "Agriculture - Crop Disease"
    ]
)

st.sidebar.info(
    f"Selected: {specialist}"
)


# ============================================================
# 6. IMAGE UPLOAD
# ============================================================

st.header("📷 Upload Image")

uploaded_file = st.file_uploader(
    "Upload a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)

uploaded_image = None

if uploaded_file is not None:

    try:

        uploaded_image = Image.open(uploaded_file)

        st.image(
            uploaded_image,
            caption="Uploaded Image",
            use_container_width=True
        )

        st.success("✅ Image uploaded successfully!")

    except Exception as e:

        st.error(
            f"❌ Could not open the image: {e}"
        )

        uploaded_image = None


# ============================================================
# 7. IMAGE ANALYSIS
# ============================================================

if uploaded_image is not None:

    if st.button(
        "🔍 Analyze Image",
        type="primary"
    ):

        st.subheader("🤖 AI Image Analysis")

        image_prompt = f"""
You are an AI educational assistant specializing in
medical and agriculture topics.

Selected specialist:
{specialist}

Analyze the uploaded image carefully.

If this is a medical image:

1. Describe visible features.
2. Explain possible observations.
3. Explain what the observations may be associated with.
4. Explain important limitations.
5. Recommend appropriate professional medical evaluation.

Do NOT:
- Claim a confirmed diagnosis.
- Pretend to be a doctor.
- Invent findings.
- State certainty when the image is unclear.

If this is an agriculture image:

1. Describe visible plant or crop features.
2. Identify possible diseases, pests, nutrient problems,
   irrigation problems, or environmental problems when supported
   by the image.
3. Explain possible causes.
4. Give general management information.
5. Give prevention suggestions.
6. Explain uncertainty if the image is insufficient.

Only describe things that can reasonably be observed.
"""

        with st.spinner("🔍 Analyzing image..."):

            result = ask_gemini(
                [
                    image_prompt,
                    uploaded_image
                ]
            )

            st.write(result)


# ============================================================
# 8. CHATBOT
# ============================================================

st.divider()

st.header("💬 Medical & Agriculture Assistant")


# ============================================================
# 9. CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

# ============================================================
# 10. VOICE INPUT AND CHAT INPUT
# ============================================================

st.subheader("🎤 Speak Your Problem")

language = st.selectbox(
    "Choose your language / మీ భాషను ఎంచుకోండి",
    ["English", "తెలుగు (Telugu)"]
)

audio = mic_recorder(
    start_prompt="🎙️ Start Recording / రికార్డింగ్ ప్రారంభించండి",
    stop_prompt="⏹️ Stop Recording / రికార్డింగ్ ఆపండి",
    key="voice_recorder"
)

if audio:
    st.audio(audio["bytes"], format="audio/wav")

    st.info(
        "Voice recorded successfully! "
        "మీ వాయిస్ రికార్డ్ అయింది!"
    )

    if st.button("📤 Send Voice / వాయిస్ పంపండి", type="primary"):
        with st.spinner("Processing your voice... / మీ వాయిస్‌ను మారుస్తున్నాం..."):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[
                        types.Part.from_bytes(
                            data=audio["bytes"],
                            mime_type="audio/wav"
                        ),
                        (
                            f"Transcribe the spoken audio in {language}. "
                            "Preserve the language spoken. "
                            "Return only the transcribed words."
                        )
                    ]
                )

                transcribed_text = (
                    response.text.strip()
                    if response.text
                    else ""
                )

                if transcribed_text:
                    st.session_state["voice_text"] = transcribed_text
                    st.success("Voice converted successfully!")
                    st.write("Your message:", transcribed_text)
                else:
                    st.warning("No speech detected. Please record again.")

            except Exception as e:
                st.error(f"Voice processing failed: {e}")


typed_message = st.chat_input(
    "Type your question / మీ ప్రశ్నను టైప్ చేయండి..."
)

if typed_message:
    user_message = typed_message
else:
    user_message = st.session_state.pop("voice_text", None)

# ============================================================
# 11. CHAT RESPONSE
# ============================================================
if user_message:

    # --------------------------------------------------------
    # Show user's message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(user_message)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )


    # --------------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an AI assistant for a website that focuses on
MEDICAL and AGRICULTURE topics.

Selected specialist:
{specialist}

User question:
{user_message}


============================================================
MEDICAL QUESTIONS
============================================================

You can answer educational questions about:

- Eye health
- Vision
- Eye symptoms
- Eye diseases
- Chest
- Lungs
- Chest conditions
- Chest X-rays
- Bone problems
- Bone X-rays
- Symptoms
- Medical terminology
- Prevention
- General healthcare
- Basic health education


For medical questions:

- Answer the user's actual question directly.
- Explain information clearly.
- Use simple language when possible.
- Do not claim a confirmed diagnosis.
- Do not pretend to replace a doctor.
- Do not invent medical findings.
- Recommend professional medical evaluation when appropriate.
- If symptoms could represent an emergency, advise the user
  to seek urgent medical care.


============================================================
AGRICULTURE QUESTIONS
============================================================

You can answer educational questions about:

- Crop diseases
- Plant diseases
- Leaf problems
- Fruit problems
- Stem problems
- Root problems
- Plant pests
- Insects
- Fungal diseases
- Bacterial diseases
- Viral plant diseases
- Soil problems
- Nutrient deficiencies
- Irrigation
- Over-irrigation
- Under-irrigation
- Fertilizers
- Crop management
- Disease prevention
- Crop care
- Pest management
- Plant growth


For agriculture questions:

- Answer the actual question directly.
- Explain possible causes.
- Explain symptoms when relevant.
- Give general management approaches.
- Give prevention suggestions.
- Explain uncertainty when information is insufficient.


============================================================
IMAGE QUESTIONS
============================================================

If an image has been uploaded and is relevant to the question:

- Use the image.
- Describe only features that can reasonably be observed.
- Do not invent details.

For medical images:

- Give educational observations.
- Do not claim a confirmed diagnosis.
- Recommend professional evaluation when appropriate.

For agriculture images:

- Describe visible crop or plant symptoms.
- Explain possible diseases, pests, nutrient deficiencies,
  irrigation issues, or environmental causes when supported.
- Explain that image-based identification may be uncertain.


============================================================
OTHER QUESTIONS
============================================================

The main purpose of this chatbot is medical and agriculture
assistance.

If a question is completely unrelated to medical or agriculture
topics, politely explain that this chatbot is designed mainly
for medical and agriculture-related questions.


============================================================
ANSWER STYLE
============================================================

Always answer the user's question clearly and directly.

Do not unnecessarily repeat these instructions.

Do not claim certainty when the available information
is insufficient.
"""


    # ========================================================
    # 12. SEND REQUEST TO GEMINI
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner("🤖 Thinking..."):

            if uploaded_image is not None:

                result = ask_gemini(
                    [
                        prompt,
                        uploaded_image
                    ]
                )

            else:

                result = ask_gemini(
                    prompt
                )

            st.write(result)


    # ========================================================
    # 13. SAVE AI RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result
        }
    )
