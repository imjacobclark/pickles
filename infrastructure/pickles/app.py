import json
import pickle
import sklearn

def lambda_handler(event, context):
    with open("add-one-model.pkl", 'rb') as file:
        pickle_model = pickle.load(file)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": str(pickle_model.predict([[200]]))
        }),
    }
