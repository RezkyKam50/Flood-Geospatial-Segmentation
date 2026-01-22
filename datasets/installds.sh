
# dummy dataset 4 testing
git xet install
git clone https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense

# real dataset Sen1Floods11 (446 samples)
aria2c -s16 -x16 https://drive.google.com/uc?id=1lRw3X7oFNq_WyzBO6uyUJijyTuYm23VS

# real dataset building damage
cd ABD
cat xview2_geotiff.tgz.a* > xview2_geotiff.tgz

# recombine 
tar -xzvf xview2_geotiff.tgz