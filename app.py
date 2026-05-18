from flask import Flask, render_template, request
from model import load_model, predict_message

app = Flask(__name__)

# Load model and metrics
model, vectorizer, accuracy, precision, recall, f1, cm = load_model()

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    message = ""

    if request.method == "POST":

        message = request.form["message"]

        result = predict_message(message, model, vectorizer)

    return render_template(
        "index.html",
        result=result,
        message=message,
        accuracy=round(accuracy,2),
        precision=round(precision,2),
        recall=round(recall,2),
        f1=round(f1,2),
        cm=cm
    )

if __name__ == "__main__":
    app.run(debug=True)