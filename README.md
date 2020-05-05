# bodypix-fun

Proof-of-Concept of FOSS Virtual Background on Node.js

## Preparing

1. Install development packages of following libraries.
   * opencv
   * gtk2 (for linux)
2. Install NPM libraries (it takes a little time to download and build bodypix, opencv...)
   ```
   npm install
   ```

## How to run

```
%  npm run ./app.js --background=/path/to/background.jpg --preview
```

You can use the composited video as a virtual webcam by
sending the displayed window to the v4l2loopback device.

```
% gst-launch-1.0 -v ximagesrc xid=$XWINDOW_ID use-damage=false ! \
    videoconvert !
    videoscale !
    "video/x-raw,width=640,height=480,framerate=30/1,format=YUY2" !
    v4l2sink device=/dev/video2
```

## References

 * [Open Source Virtual Background | BenTheElder](https://elder.dev/posts/open-source-virtual-background/)
