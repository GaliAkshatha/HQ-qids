import matplotlib.pyplot as plt

# Model results from your experiments

models = ["Classical SVM", "QSVM", "VQC"]
accuracy = [0.96, 0.913, 0.71]

plt.figure(figsize=(8,5))

plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")

plt.ylim(0,1)

for i,v in enumerate(accuracy):
    plt.text(i, v + 0.02, str(round(v,2)), ha='center')

plt.show()