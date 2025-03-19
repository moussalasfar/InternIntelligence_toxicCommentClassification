from flask import Flask, render_template, request
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("model.pkl") 
try:
    vectorizer = joblib.load("vectorizer.pkl")
    print("Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading vectorizer: {e}")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        comment = request.form["comment"]
        comment_vector = vectorizer([comment])  
        result = model.predict(comment_vector)[0]

        
        threshold = 0.5
        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        detected_labels = [labels[i] for i in range(len(result)) if result[i] > threshold]

        if detected_labels:
            prediction = f"Harmful comment detected 🚨"
        else:
            prediction = "Comment is not harmful ✅"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)




