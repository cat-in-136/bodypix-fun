const tf = require('@tensorflow/tfjs-node-gpu');
//const tf = require('@tensorflow/tfjs-node');
const bodyPix = require('@tensorflow-models/body-pix');
const cv = require('opencv4nodejs');

(async () => {
  const net = await bodyPix.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75,
    quantBytes: 2,
  });
  console.debug('BodyPix Loaded.');

  const capture = new cv.VideoCapture(0);

  const background = await cv.imreadAsync('background.jpg');
  console.time('iteration');
  {
    const cap = await capture.readAsync();
    const [ width, height ] = cap.sizes;
    await cv.imwriteAsync('capture.jpg', cap);

    console.time('bodyPix');
    const image = tf.tensor3d(await cap.getData(), [width, height, 3]);
    const { data } = await net.segmentPerson(image, {
      flipHorizontal: false,
      internalResolution: 'medium',
      segmentationThreshold: 0.7,
    });
    console.timeEnd('bodyPix');
    const mask = new cv.Mat(new Uint8Array(data), width, height, cv.CV_8U);

    const output = cap.copyTo(background, mask);
    await cv.imwriteAsync('masked.jpg', output);
  }
  console.timeEnd('iteration');

})();
