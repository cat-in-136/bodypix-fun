const tf = require('@tensorflow/tfjs-node-gpu');
//const tf = require('@tensorflow/tfjs-node');
const bodyPix = require('@tensorflow-models/body-pix');
const nodeWebcam = require('node-webcam');
const Jimp = require('jimp');

(async () => {
  const net = await bodyPix.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75,
    quantBytes: 2,
  });
  console.debug('BodyPix Loaded.');

  const webcam = nodeWebcam.create({
    saveShots: false,
    output: 'jpeg',
    device: false,
    callbackReturn: 'buffer',
    verbose: true,
  });
  const webcamCapture = () => new Promise((resolve, reject) => {
    webcam.capture('captured.jpg', (err, data) => {
      if (!!err) {
        reject(err);
      } else {
        resolve(data);
      }
    });
  });
  console.debug('Webcam opened.');

  const background = await Jimp.read('background.jpg');
  console.time('iteration');
  {
    let buf = await webcamCapture();

    console.time('bodyPix');
    const image = tf.node.decodeImage(buf);
    const { data, width, height } = await net.segmentPerson(image, {
      flipHorizontal: false,
      internalResolution: 'medium',
      segmentationThreshold: 0.7,
    });
    console.timeEnd('bodyPix');
    const mask = new Uint8Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      mask[4 * i + 0] = data[i] * 0xFF;
      mask[4 * i + 1] = data[i] * 0xFF;
      mask[4 * i + 2] = data[i] * 0xFF;
      mask[4 * i + 3] = 0xFF;
    }
    const m = await Jimp.read({data: mask, width, height});
    m.write('mask.jpg');

    const output = await Jimp.read(buf)
    await output.mask(m);
    await output.composite(background, 0, 0, { mode: Jimp.BLEND_DESTINATION_OVER });
    output.write('masked.jpg');
  }
  console.timeEnd('iteration');

})();
