from model import lg_model


def play_no_play(temp: int):
    predictions = lg_model.predict([[temp]])
    predicted_label = 'Play' if predictions == 1 else 'Not Play'
    return predicted_label
