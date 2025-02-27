
#gcloud auth login

#gcloud builds submit --tag gcr.io/testbed-452117/streamlit-tutorial-video --project=testbed-452117


gcloud builds submit --tag gcr.io/testbed-452117/streamlit-tutorial  --project=testbed-452117

gcloud run deploy --image gcr.io/testbed-452117/streamlit-tutorial' --platform managed  --project=testbed-452117 --allow-unauthenticated


gcloud builds submit --tag gcr.io/testbed-452117/spam-detector --project=testbed-452117
gcloud run deploy streamlit-spam-detector --image gcr.io/testbed-452117/spam-detector --platform managed  --allow-unauthenticated


