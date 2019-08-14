import tensorflow as tf
import matplotlib.pyplot as plt
from data import train_data, train_labels, eval_data, eval_labels
from model import cnn_model_fn, path
from sklearn.metrics import confusion_matrix

def evaltest():

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)

    return eval_results['loss']



configuration = tf.estimator.RunConfig(save_summary_steps=40,
                                           keep_checkpoint_max=1,
                                           save_checkpoints_steps=100)

classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=path,
    config= configuration)


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=500,
    num_epochs=None,
    shuffle=True)

step = 1

test_loss = []

for i in range(0, step):
  classifier.train(input_fn=train_input_fn, steps=10)
  test_loss.append(evaltest())


plt.plot(test_loss)
plt.ylabel('test_loss')
plt.xlabel('iterations')
plt.show()


def get_confusion(eval_data, eval_labels):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        shuffle=False
    )

    predictions = list(classifier.predict(input_fn=eval_input_fn))

    predicted_labels = []

    for pred in (predictions):
        v = pred["classes"]
        predicted_labels.append(v)

    print(confusion_matrix(eval_labels, predicted_labels))

    return


get_confusion(eval_data, eval_labels)

get_confusion(train_data, train_labels)


def evaluate_model(eval_data, eval_labels):

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)
    return


evaluate_model(eval_data, eval_labels)
evaluate_model(train_data, train_labels)