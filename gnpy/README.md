
Use telecominfraproject/oopt-gnpy Docker image and mount pwd in the /opt subdirectory in container.

refer to ```https://gnpy.readthedocs.io/en/master/install.html```

With ```--volume $(pwd):/opt```, ```/opt``` in the docker will be ```optical-project/gnpy/``` (this directory.)

Run ```python service-generator.py```.