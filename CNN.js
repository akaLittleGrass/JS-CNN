function CNN() {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;
  const NUM_OUTPUT_CLASSES = 10;
  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }),
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
      }),
      tf.layers.conv2d({
        kernelSize: 3,
        filters: 32,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
      }),
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
      }),
      tf.layers.flatten(),
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
      })
    ]
  });
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}
export default CNN;