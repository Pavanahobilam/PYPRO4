import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

print("Train the machine")
x = []
y = []
cap = cv2.VideoCapture(0)  # Use default camera (index 0)

while True:
    ret, frame = cap.read()
    if not ret:  # Check if cap.read() was successful
        break
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    plt.show()  # Display the frame
    ch = input("Enter l for light and d for dark and f for finish training: ")
    if ch == 'l':
        c = [(int(np.mean(frame[:, :, 0]))), (int(np.mean(frame[:, :, 1]))), (int(np.mean(frame[:, :, 2])))]
        x.append(c)
        y.append(1)
    elif ch == 'd':
        c = [(int(np.mean(frame[:, :, 0]))), (int(np.mean(frame[:, :, 1]))), (int(np.mean(frame[:, :, 2])))]
        x.append(c)
        y.append(0)
    elif ch == 'f':
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        print(x)
        print(y)
        classifier = KNeighborsClassifier()
        classifier.fit(x, y)
        break
    else:
        print('Invalid input')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Fun time")
print("Press s to check")
while True:
    ret, frame = cap.read()
    if not ret:  # Check if cap.read() was successful
        break
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    plt.show()  # Display the frame
    ch = input("Enter s to check and b to end: ")
    if ch == 's':
        c = [(int(np.mean(frame[:, :, 0]))), (int(np.mean(frame[:, :, 1]))), (int(np.mean(frame[:, :, 2])))]
        y_pred = classifier.predict([c])
        if y_pred == 1:
            print("Light")
        else:
            print("Dark")
    elif ch == 'b':
        break
    else:
        print('Invalid input')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
