# HTUpath-finder
import pandas as pd

# 1) اقرأ الملفات
df1 = pd.read_csv(r"C:\Users\ranee\Downloads\Results_with_Label.csv")
df2 = pd.read_csv(r"C:\Users\ranee\Downloads\htu_majors_interests.csv")

# (مهم) اسم عمود التخصص بالملف الثاني (آخر عمود غالباً اسمه label)
label_col = df2.columns[-1]

# 2) جهّز df1 ليصير نفس شكل df2 (أول 34 عمود)
df1_ready = df1.iloc[:, :34].copy()
df1_ready.columns = df2.columns[:34]

# 3) طلع التخصص (Label) من آخر 3 أعمدة في df1
def get_label(i):
    current = str(df1.iloc[i, -3]).strip()
    status  = str(df1.iloc[i, -2]).strip().lower()
    pref    = str(df1.iloc[i, -1]).strip()

    if any(x in status for x in ["yes", "happy", "n/a", "نعم"]):
        return current
    else:
        return pref

df1_ready[label_col] = [get_label(i) for i in range(len(df1_ready))]

# (اختياري بس مهم للتطابق) شيل المسافات من التخصصات
df1_ready[label_col] = df1_ready[label_col].astype(str).str.strip()
df2[label_col]       = df2[label_col].astype(str).str.strip()

# 4) الصق df1 تحت df2
combined = pd.concat([df2, df1_ready], ignore_index=True)

# 5) رتّب الأعمدة نفس df2 (عشان يصير التطابق مضبوط)
combined = combined[df2.columns]

# 6) احذف التكرار (تطابق كامل من أول عمود لآخر عمود)
final_clean = combined.drop_duplicates(keep="first")

# 7) اطبع النتائج
print(" df2 rows:", len(df2))
print(" added rows:", len(df1_ready))
print(" after removing exact duplicates:", len(final_clean))

print("\nTop 10 labels:")
print(final_clean[label_col].value_counts().head(10))
final_clean.to_csv(
    r"C:\Users\ranee\Downloads\htu_data_clean_v2 (2).csv",
    index=False
)
import numpy as np
import pandas as pd
import nltk
import speech_recognition as sr
import pyttsx3
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from nltk.probability import FreqDist
from transformers import pipeline

nltk.download("punkt")
nltk.download("punkt_tab")

# ================== VOICE ==================
engine = pyttsx3.init()
engine.setProperty("rate", 165)
engine.setProperty("volume", 1.0)

def speak(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 1.2

NUM_MAP = {
    "1":"1","one":"1","wahed":"1","واحد":"1","recommendation":"1",

    "2":"2","two":"2","ithnain":"2","اثنين":"2","eligibility":"2",

    "3":"3","three":"3","ثلاثة":"3","advisor":"3","full advisor":"3",

    "4":"4","four":"4","اربعة":"4","question":"4","questions":"4",

    "5":"5","five":"5","exit":"5","quit":"5","خروج":"5"
}

def normalize_number(txt):
    txt = txt.lower()
    for k,v in NUM_MAP.items():
        if k in txt:
            return v
    return None

def voice_input(prompt):
    speak(prompt)
    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            num = normalize_number(text)
            return num if num else text.lower()
        except:
            speak("Please repeat clearly")

def read_number(prompt):
    speak(prompt)
    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5)
            return float(recognizer.recognize_google(audio))
        except:
            speak("Say a number clearly")

# ================== FILES ==================
QUESTIONS_FILE = r"C:\Users\ranee\Downloads\HTU_Survey_Questions_Template.csv"
DATASET_FILE   = r"C:\Users\ranee\Downloads\htu_data_clean_v2.csv"
POLICY_FILE    = r"C:\Users\ranee\Downloads\labeled_chunks_Admission _Policy.csv"

# ================== MAJORS ==================
ENGINEERING = {
    "Mechanical Engineering","Energy Engineering","Electrical Engineering",
    "Architecture","Cybersecurity Engineering","Civil Engineering"
}
CS_GROUP = {
    "Industrial Engineering","Computer Science","Data Science","Artificial Intelligence"
}
GAME = {"Game Design and Development"}

YES_SET = {"y","yes","نعم","اه"}
NO_SET  = {"n","no","لا"}

# ================== INTEREST ==================
def load_questions():
    df = pd.read_csv(QUESTIONS_FILE, header=None)
    q=[]
    for i in range(1,len(df)):
        f = str(df.iloc[i,1]).strip()
        qtxt = str(df.iloc[i,2]).strip()
        if f and qtxt and f.lower()!="nan" and qtxt.lower()!="nan":
            q.append((f,qtxt))
    return q

def interest_recommendation():
    speak("Answer the questions by typing yes or no")
    questions = load_questions()
    student_answers = {}

    for f,q in questions:
        while True:
            ans = input(f"{q} (y/n): ").strip().lower()

            if ans in YES_SET:
                student_answers[f] = 1
                break
            elif ans in NO_SET:
                student_answers[f] = 0
                break
            else:
                print("Only y/n allowed")

    df = pd.read_csv(DATASET_FILE)
    label = df.columns[-1]
    features = df.columns[:-1]

    vec = np.array([student_answers.get(c,0) for c in features]).reshape(1,-1)
    sims = cosine_similarity(vec, df[features].values)[0]
    df["sim"]=sims

    top = df.groupby(label)["sim"].max().sort_values(ascending=False).head(2)

    speak("Top recommended majors based on your interests are")
    for m in top.index:
        speak(m)

    return set(top.index)

# ================== ELIGIBILITY ==================
def eligibility_check():
    speak("Choose certificate type")
    speak("1 Tawjihi 2 IGCSE 3 IB 4 American 5 BTEC")

    while True:
        c = voice_input("Say your choice")
        if c in {"1","2","3","4","5"}:
            break
        speak("Say a valid number")

    eligible = {}

    def add(majors, degree):
        for m in majors:
            eligible[m] = degree

    if c=="1":
        m=read_number("Math percentage")
        p=read_number("Physics percentage")
        o=read_number("Overall percentage")
        if m>=80 and p>=80 and o>=80: add(ENGINEERING,"Bachelor")
        if m>=70 and p>=70 and o>=70: add(ENGINEERING,"Technical")
        if m>=85 and p>=85 and o>=85: add(CS_GROUP,"Bachelor")
        if m>=70 and p>=70 and o>=70: add(CS_GROUP,"Technical")
        if m>=70 and p>=70 and o>=70: add(GAME,"Technical")

    if c=="2":
        m=read_number("Math grade 1-9")
        p=read_number("Physics grade 1-9")
        ol=read_number("Physics OL grade 1-9")
        o=read_number("Overall percentage")
        if m>=6 and p>=6 and o>=80 and ol>=6: add(ENGINEERING,"Bachelor")
        if m>=5 and p>=5 and o>=70 and ol>=4: add(ENGINEERING,"Technical")
        if m>=6 and p>=6 and o>=85 and ol>=6: add(CS_GROUP,"Bachelor")
        if m>=5 and p>=5 and o>=70 and ol>=4: add(CS_GROUP,"Technical")
        if m>=5 and p>=5 and o>=70 and ol>=4: add(GAME,"Technical")

    if c=="3":
        m=read_number("Math score")
        p=read_number("Physics score")
        o=read_number("Overall percentage")
        if m>=5 and p>=5 and o>=80: add(ENGINEERING,"Bachelor")
        if m>=4 and p>=4 and o>=70: add(ENGINEERING,"Technical")
        if m>=5 and p>=5 and o>=85: add(CS_GROUP,"Bachelor")
        if m>=4 and p>=4 and o>=70: add(CS_GROUP,"Technical")
        if m>=4 and p>=4 and o>=70: add(GAME,"Technical")

    if c=="4":
        s=read_number("ACT or SAT or AP score")
        o=read_number("Overall percentage")
        if s>=28 and o>=80: add(ENGINEERING,"Bachelor")
        if s>=24 and o>=70: add(ENGINEERING,"Technical")
        if s>=28 and o>=85: add(CS_GROUP,"Bachelor")
        if s>=24 and o>=70: add(CS_GROUP,"Technical")
        if s>=24 and o>=70: add(GAME,"Technical")

    if c=="5":
        m=read_number("Math level 1 Merit 2 Distinction")
        p=read_number("Physics level 1 Merit 2 Distinction")
        o=read_number("Overall percentage")
        if m>=2 and p>=2 and o>=80: add(ENGINEERING,"Bachelor")
        if m>=1 and p>=1 and o>=70: add(ENGINEERING,"Technical")
        if m>=2 and p>=2 and o>=85: add(CS_GROUP,"Bachelor")
        if m>=1 and p>=1 and o>=70: add(CS_GROUP,"Technical")
        if m>=1 and p>=1 and o>=70: add(GAME,"Technical")

    speak("Eligibility results")
    for m,d in eligible.items():
        speak(f"{m} {d}")

    if not eligible:
        speak("You are not eligible for any program")

    return eligible

# ================== FULL ADVISOR ==================
def full_advisor():
    interest = interest_recommendation()
    eligible = eligibility_check()

    final = interest & set(eligible.keys())

    if final:
        speak("Best majors for you based on interest and eligibility")
        for m in final:
            speak(f"{m} {eligible[m]}")
    else:
        speak("No match between interest and eligibility")

# ================== RAG ==================
policy_df = pd.read_csv(POLICY_FILE)
policy_df["clean_text"] = policy_df["text"].astype(str).str.strip()
policy_df = policy_df[policy_df["clean_text"].str.len()>0]

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def retrieve_chunks(question, df, k=3):
    all_text = " ".join(df["clean_text"])
    vocab = sorted(set(word_tokenize(all_text.lower())))

    def bow(text):
        tokens = word_tokenize(text.lower())
        freq = FreqDist(tokens)
        return np.array([freq.get(w,0) for w in vocab])

    qv = bow(question)
    sims = []
    for t in df["clean_text"]:
        tv = bow(t)
        sim = np.dot(qv,tv)/(np.linalg.norm(qv)*np.linalg.norm(tv)+1e-9)
        sims.append((sim,t))

    sims.sort(reverse=True)
    return [x[1] for x in sims[:k]]

def rag_mode():
    speak("Question answering mode started")
    while True:
        q = input("Ask your question (or exit): ")
        if q.lower() in ["exit","quit"]:
            break
        chunks = retrieve_chunks(q, policy_df)
        context = "\n".join(chunks)[:2000]
        result = qa_pipeline(question=q, context=context)
        speak(result["answer"])
        print("Evidence:", chunks[0][:300])

# ================== MENU ==================
def menu():
    print("\n1 Recommendation")
    print("2 Eligibility")
    print("3 Full Advisor")
    print("4 Questions (RAG)")
    print("5 Exit")

# ================== MAIN ==================
def main():
    speak("Welcome to HTU PathFinder")
    while True:
        menu()
        c = voice_input("Say your choice")

        if c=="1": interest_recommendation()
        elif c=="2": eligibility_check()
        elif c=="3": full_advisor()
        elif c=="4": rag_mode()
        elif c=="5":
            speak("Goodbye")
            break
        else:
            speak("Say a valid option")

if __name__=="__main__":
    main()

