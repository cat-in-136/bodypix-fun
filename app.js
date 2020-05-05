const argv = require('yargs')
  .usage('Usage: $0 [options]')
  .option('input', {
    alias: 'i',
    default: 0,
    coerce: (arg) => {
      if (/^[0-9]+$/.test(arg)) {
        return parseInt(arg);
      } else {
        return arg;
      }
    },
    desc: 'input file path or camera capture (number)',
  })
  .option('background', {
    alias: 'b',
    desc: 'background file',
    type: 'string',
  })
  .option('resize', {
    alias: 's',
    coerce: (arg) => {
      if (/^([0-9]+)x([0-9]+)$/.test(arg)) {
        return {width: parseInt(RegExp.$1), height: parseInt(RegExp.$2)};
      } else {
        throw new Error("specify image size e.g. -s 640x480");
      }
    },
    desc: 'resize output image size e.g. 640x480',
    type: 'string',
  })
  .option('output', {
    alias: 'o',
    desc: 'output file path',
    type: 'string',
  })
  .option('preview', {
    desc: 'preview on window',
    type: 'bool',
  })
  .check((args) => {
    // https://github.com/yargs/yargs/issues/1318#issuecomment-568301488
    const arrayArgs = Object.entries(args)
      .filter(([k, v]) => typeof k === 'string' && /[A-Z]+/i.test(k) && Array.isArray(v))
      .map(([k, _]) => k)

    return arrayArgs.length > 0
      ? `Too many arguments: ${arrayArgs.join(', ')}`
      : true
  })
  .help()
  .alias('help', 'h')
  .version(false)
  .argv;

const tf = require('@tensorflow/tfjs-node-gpu');
//const tf = require('@tensorflow/tfjs-node');
const bodyPix = require('@tensorflow-models/body-pix');
const cv = require('opencv4nodejs');

(async () => {
  const vCap = new cv.VideoCapture(argv['input']);
  const vCapSize = new cv.Size(
    vCap.get(cv.CAP_PROP_FRAME_WIDTH),
    vCap.get(cv.CAP_PROP_FRAME_HEIGHT)
  );
  const {width, height} = (!!argv['resize']) ? argv['resize'] : vCapSize;

  const net = await bodyPix.load({
    architecture: 'MobileNetV1',
    outputStride: 16,
    multiplier: 0.75,
    quantBytes: 2,
  });
  console.debug('BodyPix Loaded.');

  const background = (!!argv['background']) ?
    cv.imread(argv['background']).resize(height, width) :
    new cv.Mat(height, width, cv.CV_8UC3, [0, 255, 0]);

  const out = (!!argv['output'])? new cv.VideoWriter(
    argv['output'],
    cv.VideoWriter.fourcc('h264'), //vCap.get(cv.CAP_PROP_FOURCC),
    vCap.get(cv.CAP_PROP_FPS),
    new cv.Size(width, height),
    true) : null;

  try {
    while (true) {
      let frame = vCap.read();
      if (frame.empty) { break; }
      frame = frame.resize(height, width);

      console.time('bodyPix');
      const image = tf.tensor3d(await frame.getData(), [height, width, 3]);
      const { data } = await net.segmentPerson(image, {
        flipHorizontal: false,
        internalResolution: 'medium',
        segmentationThreshold: 0.7,
      });
      image.dispose();
      console.timeEnd('bodyPix');
      const mask = new cv.Mat(new Uint8Array(data), height, width, cv.CV_8U);

      const outFrame = new cv.Mat(height, width, frame.type);
      background.copyTo(outFrame);
      frame.copyTo(outFrame, mask);
      if (!!out) {
        out.write(outFrame);
      }

      if (argv['preview']) {
        cv.imshow('out', outFrame);
        if (cv.waitKey(1) == 'q'.charCodeAt(0)) {
          break;
        }
      }
    }
  } finally {
    vCap.release();
    if (!!out) {
      out.release();
    }

    if (argv['preview']) {
      cv.destroyAllWindows();
    }
  }
})();
