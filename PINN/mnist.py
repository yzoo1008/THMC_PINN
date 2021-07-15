import tensorflow as tf
import numpy as np

print(tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.min(), x_train.max())
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

print(x_train.shape, x_test.shape)

# Dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Model
input_ = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, 3, activation='relu')(input_)
x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output_ = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.models.Model(input_, output_)
model.summary()

# Loss
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

# Optimizer
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()

test_loss = tf.keras.metrics.Mean()
test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(images, labels):
    # 미분을 위한 GradientTape을 적용합니다.
    with tf.GradientTape() as tape:
        # 1. 예측 (prediction)
        predictions = model(images)
        # 2. Loss 계산
        loss = loss_function(labels, predictions)

    # 3. 그라디언트(gradients) 계산
    gradients = tape.gradient(loss, model.trainable_variables)

    # 4. 오차역전파(Backpropagation) - weight 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # loss와 accuracy를 업데이트 합니다.
    train_loss(loss)
    train_acc(labels, predictions)


@tf.function
def test_step(images, labels):
    # 1. 예측 (prediction)
    predictions = model(images)
    # 2. Loss 계산
    loss = loss_function(labels, predictions)

    # Test셋에 대해서는 gradient를 계산 및 backpropagation 하지 않습니다.

    # loss와 accuracy를 업데이트 합니다.
    test_loss(loss)
    test_acc(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = '에포크: {}, 손실: {:.5f}, 정확도: {:.2f}%, 테스트 손실: {:.5f}, 테스트 정확도: {:.2f}%'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_acc.result()*100,
                           test_loss.result(),
                           test_acc.result()*100))
