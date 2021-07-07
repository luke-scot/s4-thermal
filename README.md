# s4-thermal
Emulating satellite thermal imagery

### Camera Info

Camera: FLIR VUE PRO

sensor format: 640x512 pixels - each pixel is 17um square

Focal length of camera lens is 9mm

FOV is 69 deg x 56 deg

At 120m altitude the nadir has an image scale of about 23cm per pixel

Across the image the scale is on average about 27cm per pixel

An image is about 173m x 138m on the ground.

### Converting RGB images to thermal 
If unsuccesful in getting RGB_to_temp to run due to package issues, do not fear! You can run a whole directory through the flir-image-extractor CLI.
Follow https://github.com/nationaldronesau/FlirImageExtractor for details but here is the low down for Windows machines.

1. To install exiftool on Windows for use in this CLI, download the exiftool windows executable from here. Extract exiftool(-k).exe and rename to exiftool.exe. Copy this executable to C:\Windows on your computer. You will need admin permissions to do this. Doing this will make exiftool available to the CLI.

2. python -m pip install flir-image-extractor-cli

3. flir-image-extractor-cli

4. Pop the directory path in and enjoy the ride.
