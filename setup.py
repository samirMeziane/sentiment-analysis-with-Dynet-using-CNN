import setuptools



with open("README.md", "r") as fh:

    long_description = fh.read()





setuptools.setup(

     name='cnn_text_classification',

     version='4.0',

     scripts=['predict_cnn.py','train_cnn.py'] ,

     author="Wissam MK",

     author_email="mk.wissam.92@gmail.com",

     description="implementation of the paper Char CNN",

     long_description=long_description,

   long_description_content_type="text/markdown",

     #url="https://github.com/javatechy/dokr",

     packages=setuptools.find_packages(),

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

)