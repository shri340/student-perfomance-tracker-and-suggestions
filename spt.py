import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Student Predictor",
    page_icon="🎓",
    layout="wide"
)

# ---------------- DARK MODE ----------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

bg_color = "#0E1117" if dark_mode else "#F5F7FA"
text_color = "white" if dark_mode else "black"

st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}
.title {{
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #4A90E2;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🎓 AI Student Performance Predictor</div>', unsafe_allow_html=True)
st.write("### 📊 Predict your performance and get smart suggestions")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📥 Enter Student Details")

attendance = st.sidebar.slider("Attendance (%)", 0, 100, 75)
study_hours = st.sidebar.slider("Study Hours", 0, 10, 3)
assignments = st.sidebar.slider("Assignments Completed", 0, 10, 5)
previous_marks = st.sidebar.slider("Previous Marks (%)", 0, 100, 70)

# ---------------- DATASET ----------------
data = pd.DataFrame({
    'attendance': [80, 60, 90, 50, 70, 85, 40],
    'study_hours': [3, 1, 4, 1, 2, 5, 1],
    'assignments': [5, 2, 6, 1, 3, 7, 1],
    'previous_marks': [75, 50, 88, 40, 65, 90, 35],
    'result': [1, 0, 1, 0, 1, 1, 0]
})

X = data[['attendance', 'study_hours', 'assignments', 'previous_marks']]
y = data['result']

model = RandomForestClassifier()
model.fit(X, y)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Analytics", "💡 Suggestions"])

# ================= TAB 1 =================
with tab1:
    st.subheader("🎯 Performance Prediction")

    if st.button("🚀 Predict Now"):

        prediction = model.predict([[attendance, study_hours, assignments, previous_marks]])

        score = (
            attendance * 0.3 +
            study_hours * 10 * 0.2 +
            assignments * 10 * 0.2 +
            previous_marks * 0.3
        )

        st.progress(int(score))

        if prediction[0] == 1:
            st.success("🎉 You are likely to PASS!")
        else:
            st.error("⚠️ Risk of FAIL. Improve performance.")

        st.info(f"📈 Performance Score: {int(score)} / 100")

# ================= TAB 2 =================
with tab2:
    st.subheader("📊 Performance Analytics")

    labels = ['Attendance', 'Study Hours', 'Assignments', 'Previous Marks']
    values = [attendance, study_hours * 10, assignments * 10, previous_marks]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Student Performance Breakdown")

    st.pyplot(fig)

    st.write("### 📉 Comparison with Average Student")

    avg = [70, 30, 50, 65]
    your = values

    fig2, ax2 = plt.subplots()
    x = np.arange(len(labels))

    ax2.bar(x - 0.2, avg, 0.4, label="Average")
    ax2.bar(x + 0.2, your, 0.4, label="You")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()

    st.pyplot(fig2)

# ================= TAB 3 =================
with tab3:
    st.subheader("💡 Personalized Suggestions")

    if attendance < 75:
        st.warning("📌 Improve attendance to at least 75%")
    else:
        st.success("✅ Attendance is good")

    if study_hours < 2:
        st.warning("📌 Study at least 2–3 hours daily")
    else:
        st.success("✅ Study habits are good")

    if assignments < 5:
        st.warning("📌 Complete more assignments")
    else:
        st.success("✅ Assignment performance is good")

    if previous_marks < 60:
        st.warning("📌 Focus on improving conceptual understanding")
    else:
        st.success("✅ Previous academic performance is strong")

    st.markdown("---")
    st.write("### 🧠 AI Advice")

    overall = (attendance + study_hours*10 + assignments*10 + previous_marks) / 4

    if overall < 50:
        st.error("⚠️ High risk: Immediate improvement needed")
    elif overall > 75:
        st.success("🎯 Excellent performance trend!")
    else:
        st.info("📈 Moderate performance — keep improving steadily")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("💻 Advanced AI Project | Built using Python & Streamlit")
