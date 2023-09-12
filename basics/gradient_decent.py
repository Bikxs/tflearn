import tensorflow as tf


@tf.function
def custom_xor(a, b, c):
    return tf.math.minimum(tf.math.minimum(a, b), c)

if __name__ == '__main__':
    print("Gradient decent and gradient tape for tensorflow booleans")
    a = tf.constant(True,dtype=tf.bool,name='a')
    b = tf.zeros(1,dtype=tf.bool,name='b')
    c = tf.constant(False,dtype=tf.bool,name='c')
    y= tf.constant(False,dtype=tf.bool,name='y')
    with tf.GradientTape() as tape:
        y_pred =custom_xor(a,b,c)
        loss = tf.keras.losses.binary_crossentropy(y_true=tf.cast(y, tf.float32), y_pred=tf.cast(y_pred, tf.float32))
    
    
    gradients = tape.gradient(loss, a)
    print('loss',loss)
    print('gradients',gradients)
    print('y',y,'y_pred',y_pred)