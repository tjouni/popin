Backend is deployed as an AWS Lambda using a Docker image with Serverless.
To deploy the code:

1. Change the S3 bucket and file name on `backend.py:44`.
2. Change the region and S3 resource on `serverless.yml`.
3. Install Serverless with `npm install -g serverless`
4. Set AWS credentials as documented on [Serverless documentation](https://www.serverless.com/framework/docs/providers/aws/guide/credentials)
5. Run `sls deploy`

For local usage, modify the function `handler` on `backend.py` to open a locally stored image file, and class `Predictor` to use locally stored .ckpt weights.
