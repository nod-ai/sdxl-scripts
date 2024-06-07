# Validating accuracy

In order to validate the accuracy of the UNet or Scheduled-UNet model, you can either generate an image using the entire pipeline and compare against the reference image or you can compute the difference between the output of the UNet and a reference Pytorch model.

## Generating an image

We can use the ``run-txt2img-python.sh`` script in the sdxl-scripts repo to generate the image. Prior to doing that, we need to clone the SHARK-Turbine repo and create a python virtualenv with all the SHARK-Turbine dependencies. After we activate the venv, we can run the script as below. Before running the script, rename the scheduled_unet.vmfb in your tmp/ directory to PNDM_unet_30.vmfb when using the PNDM scheduler.
```
./run-txt2img-python.sh [batch coount] [device id] [weigths path]
```
For evaluating accuracy, we can set batch count to 1, device id to any GPU id and the weights path on the perf machine is ```/data/shark```. You can compare the image you get with the reference image below.
![Reference Image](./sdxl_output_2024-06-05_17-52-37_0_0.png)
