import gradio as gr
import pandas as pd
import joblib as jb

# Load the model once, when the app starts
random_forest_model = jb.load("models/titanic_forest_model")

# PILLAR1: THE FUNCTION


def predict_survival(name, Pclass, Sex, Age, SibSp, PaCh, Fare):
    """
    Predict survival using our model
    """
    input_data = pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [Sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'PaCh': [PaCh],
        'Fare': [Fare]
    })

    prediction = random_forest_model.predict(input_data)[0]
    probabilities = random_forest_model.predict_proba(input_data)[
        0]  # to explain
    if prediction == 0:
        result = f'Wooo 😮 {name.title()}, you Died 😢'
    else:
        result = f'Cheers 🥂 {name.title()}, you Survived 😁'

    # gr.label expects a dict of {class name: probability}
    probs = {"Died": float(probabilities[0]),
             "Survived": float(probabilities[1])}
    return result, probs


# PILLAR 2 : THE INPUTS (one per function parameter, same order)
inputs = [
    gr.Textbox(label="Name", placeholder="Enter your name"),
    gr.Radio([1, 2, 3], label="Passenger Class (1=First)", value=3),
    gr.Radio([("Male", 0), ("Female", 1)], label="Sex", value=0),
    gr.Slider(0, 80, value=30, step=1, label="Age"),
    gr.Number(label="Siblings/Spouses aboard", value=0, precision=0),
    gr.Number(label="Parents/Child aboard", value=0, precision=0),
    gr.Number(label="Fare paid ($)", value=30)
]

# PILLAR 3 : THE OUTPUTS (one per function return value, same order)
outputs = [
    gr.Textbox(label="Prediction"),
    gr.Label(label="Probability"),
]

# Wire the three pillars together
demo = gr.Interface(
    fn=predict_survival,
    inputs=inputs,
    outputs=outputs,
    title="Titanic Survival Prediction",
    description="Enter a passenger's details to predict if our random forest thinks they'd survived the Titanic disaster.",
    examples=[
        ["Musa Adam", 3, 0, 20, 0, 0, 8],
        ["Ayo Tofunmi", 1, 1, 17, 0, 1, 150],
    ],
)

demo.launch()
