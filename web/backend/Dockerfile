FROM public.ecr.aws/lambda/python:3.8

COPY ./ ${LAMBDA_TASK_ROOT}

RUN python3 -m pip install -r requirements.txt --targer ${LAMBDA_TASK_ROOT}

CMD ["backend.handler"]