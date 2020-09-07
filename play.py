import cv2
from keras.models import load_model
import numpy as np
from random import choice
import imutils

CLASS_REV_MAP = {
    0: "rock",
    1: "paper",
    2: "scissor",
    3: "none"
}

def mapper(index):
    return CLASS_REV_MAP[index]

def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"
    if move1 == "rock":
        if move2 == "scissor":
            return "User"
        if move2 == "paper":
            return "Computer"
    if move1 == "scissor":
        if move2 == "rock":
            return "User"
        if move2 == "paper":
            return "User"
    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissor":
            return "Computer"
model = load_model("rock_paper_scissor.h5")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

prev_move = None

while True:
    ret, frame = cap.read()
    # frame = imutils.resize(frame, width=1920)
    if not ret:
        continue

    # user's space
    cv2.rectangle(frame, (100,100), (500,500), (255, 255, 255), 2)

    # computer's space
    cv2.rectangle(frame,(800,100), (1200,500), (255,255,255), 2)

    # extracting user's image
    ui = frame[100:500, 100:500]
    img = cv2.cvtColor(ui, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227,227))

    # predicting the move made
    pred = model.predict_classes(np.array([img]))
    user_move = mapper(pred[0])

    if prev_move != user_move:
        if user_move != "none":
            computer_move = choice(['rock','paper','scissor'])
            winner = calculate_winner(user_move,computer_move)
        else:
            computer_move = "none"
            winner = "Waiting..."
    prev_move = user_move

    # displaying the information
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame,"Your Move: " + user_move, (50, 50), font, 1.2, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move, (750, 50), font, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner, (400, 600), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

    if computer_move != 'none':
       icon = cv2.imread("icons/{}.png".format(computer_move))
       icon = cv2.resize(icon,(400,400))
       frame[100:500, 800:1200] = icon

    cv2.imshow("Rock Paper Scissor", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
