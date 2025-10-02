**WIP**

to use the model add a folder inside Segmentation_Models called data and include the BraTS 2016 training and validation data.
Please download it from: [https://www.med.upenn.edu/sbia/brats2016.html](https://www.med.upenn.edu/sbia/brats2016.html)

to run the backend, create a .env file in the backend folder with the following environment variables:

# PostgreSQL

DB_HOST
DB_NAME
DB_USER
DB_PASSWORD
DB_PORT

# S3

AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
S3_BUCKET
