# Sanity check
import println as p
p.Logger.log("hello, world!")
#%%
# Get the data
import image_process as ip
process = ip.ProcessFlowers()
process.purpose()
process.write_source_data("https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz")
# if using debugger, we can visualize the data we just got
process.displayFirstImage('roses/*', 0)
process.displayFirstImage('roses/*', 1)

# Start uploading to keras to do image classification
process.load()

process.visualize()

process.normalize()
process.tune()

# %%
